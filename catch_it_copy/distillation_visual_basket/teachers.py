"""Frozen teacher restoration and parity checks against original inference methods."""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from distillation.teachers import Teacher, ROOT, sha256, original_method
from distillation.core import observation_array, ActionAdapter
from .core import action_mask


def resolve(path):
    p=Path(path)
    return p if p.is_absolute() else ROOT/p


def basket_observation(obs):
    out=np.concatenate([observation_array(obs),obs['basket']['rel_pos3d'],obs['basket']['target_rel_pos2d'],obs['mobile']],axis=-1).astype(np.float32)
    if out.shape[-1]!=51 or not np.isfinite(out).all():raise ValueError('Expected Basket raw51')
    return out


def basket_method(name):
    path=ROOT/'gym_dcmm/algs/ppo_dcmm/ppo_dcmm_catch_one_stage.py'
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PPO_Catch_OneStage')
    node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
    module=ast.Module(body=[node],type_ignores=[]);ns={'torch':torch,'np':np}
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),ns)
    return ns[name]


class FrozenTeacher:
    def __init__(self,task,cfg,device):
        self.task=task;self.device=torch.device(device);self.cfg=cfg
        self.path=resolve(cfg['checkpoint']);self.hash=sha256(self.path)
        if self.hash!=cfg['sha256']:raise ValueError('Teacher SHA256 mismatch: '+str(self.path))
        if task!='basket':
            self.teacher=Teacher(dict(cfg,checkpoint=str(self.path)),device)
            self.saved=None
        else:
            from gym_dcmm.algs.ppo_dcmm.basket_mobile_joint import JointActorCritic,JointNormalizer
            saved=torch.load(self.path,map_location='cpu',weights_only=True);self.saved=saved
            required=dict(joint_actor_version='phase_conditioned_base2_arm18_v1',action_layout='joint_base2_arm18_phase_v1',
                          basket_training_stage='mobile_joint',basket_observation_version='mobile51_v1',basket_arm_control='joint_velocity',basket_reward_version='basket_mobile_v1')
            if any(saved.get(k)!=v for k,v in required.items()):raise ValueError('Unsupported Basket teacher contract')
            for key in ('basket_mobile_options','basket_finger_options','basket_carry_options'):
                expected=dict(saved[key]);actual=dict(cfg['env_kwargs'][key])
                if key=='basket_mobile_options':
                    # Deliberate mobile-only student reset distribution; controller dynamics unchanged.
                    expected['fixed_throw_probability']=0.
                if expected!=actual:raise ValueError('Basket environment/checkpoint options differ: '+key)
            state=saved['model'];units=[v.shape[0] for k,v in state.items() if k.startswith('navigation.actor_mlp.mlp.') and k.endswith('weight')]
            self.model=JointActorCritic(units,conditioned=False).to(device)
            self.normalizer=JointNormalizer(conditioned=False).to(device)
            self.model.load_state_dict(state,strict=True);self.normalizer.load_state_dict(saved['running_mean_std'],strict=True)
            for module in (self.model,self.normalizer):
                module.eval().requires_grad_(False)
                if not all(torch.isfinite(v).all() for v in module.state_dict().values()):raise ValueError('Non-finite Basket teacher')

    @torch.no_grad()
    def predict(self,obs):
        if self.task!='basket':return self.teacher.predict(observation_array(obs))
        raw=torch.as_tensor(basket_observation(obs),device=self.device)
        return self.model.act_inference({'obs':self.normalizer(raw)}).clamp(-1,1).cpu().numpy()

    def parity(self):
        rng=np.random.default_rng(908)
        raw=rng.normal(size=(12,30)).astype(np.float32)
        obs=dict(base={'v_lin_2d':raw[:,:2]},arm=dict(ee_pos3d=raw[:,2:5],ee_quat=raw[:,5:9],ee_v_lin_3d=raw[:,9:12]),object=dict(pos3d=raw[:,12:15],v_lin_3d=raw[:,15:18]),hand=raw[:,18:])
        adapter=ActionAdapter(self.cfg)
        if self.task!='basket':
            original=original_method(self.teacher.family/'ppo_dcmm_catch_two_stage.py','obs2tensor')
            np.testing.assert_allclose(original(SimpleNamespace(device=self.device),obs).cpu().numpy(),observation_array(obs),atol=1e-7)
            inference=self.teacher.parity_check(raw)
            native=self.predict(obs)
            proxy=SimpleNamespace(env=SimpleNamespace(call=lambda _:['Catching']),action_catch_denorm=self.cfg['action_scales'])
            reference=original_method(self.teacher.family/'ppo_dcmm_catch_two_stage.py','action2dict')(proxy,torch.from_numpy(native))
        else:
            obs['basket']=dict(rel_pos3d=rng.normal(size=(12,3)).astype(np.float32),target_rel_pos2d=rng.normal(size=(12,2)).astype(np.float32))
            obs['mobile']=rng.random((12,16),dtype=np.float32);obs['mobile'][:,12]=np.arange(12)%2
            proxy=SimpleNamespace(device=self.device,model=self.model,running_mean_std=self.normalizer,mobile_basket=True,fixed_basket=False,
                                  basket_training_stage='mobile_joint',mobile_base_scale=self.cfg['action_scales'][0],
                                  action_catch_denorm=self.cfg['action_scales'],env=SimpleNamespace(call=lambda _:['Catching']))
            raw=basket_observation(obs)
            np.testing.assert_allclose(basket_method('obs2tensor')(proxy,obs).cpu().numpy(),raw,atol=1e-7)
            with torch.no_grad():reference_prediction=basket_method('model_act')(proxy,{'obs':torch.as_tensor(raw,device=self.device)},inference=True)['actions'].clamp(-1,1).cpu().numpy()
            native=self.predict(obs);inference=float(np.max(np.abs(native-reference_prediction)))
            reference=basket_method('action2dict')(proxy,torch.from_numpy(native))
            self.model.eval();self.normalizer.eval()
            # Explicitly verify the phase contracts for both halves of the synthetic batch.
            assert np.all(native[::2,2:]==0) and np.all(native[1::2,:2]==0)
        if inference>1e-5:raise ValueError('Inference mismatch')
        command=adapter.canonical(native)
        for key,value in adapter.action_dict(command).items():np.testing.assert_allclose(value,reference[key],atol=1e-7)
        return dict(sha256=self.hash,inference_max_error=inference,observation_parity=True,physical_action_parity=True)
