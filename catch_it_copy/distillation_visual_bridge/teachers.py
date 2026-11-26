"""Frozen teacher restoration and parity checks against original inference methods."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from distillation.teachers import Teacher, ROOT, sha256, original_method
from distillation.core import observation_array, ActionAdapter


def resolve(path):
    p=Path(path)
    return p if p.is_absolute() else ROOT/p


class FrozenTeacher:
    def __init__(self,task,cfg,device):
        self.task=task;self.device=torch.device(device);self.cfg=cfg
        self.path=resolve(cfg['checkpoint']);self.hash=sha256(self.path)
        if self.hash!=cfg['sha256']:raise ValueError('Teacher SHA256 mismatch: '+str(self.path))
        if task!='basket':
            self.teacher=Teacher(dict(cfg,checkpoint=str(self.path)),device)
            self.saved=None
        else:
            from basket_mobile_bridge.policy import Policy, SCHEMA
            saved=torch.load(self.path,map_location='cpu',weights_only=False);self.saved=saved
            if saved.get('schema')!=SCHEMA:raise ValueError('Expected Basket mobile bridge teacher')
            expected=dict(saved['config']['options']);actual=dict(cfg['env_kwargs']['options'])
            for key in ('evaluation_task','basket_diameter_m'):actual.pop(key,None)
            if actual!=expected:raise ValueError('Bridge teacher options differ from checkpoint')
            self.model=Policy().to(device)
            self.model.load_state_dict(saved['model'],strict=True)
            self.model.eval().requires_grad_(False)
            if not all(torch.isfinite(v).all() for v in self.model.state_dict().values()):raise ValueError('Non-finite bridge teacher')

    @torch.no_grad()
    def predict(self,obs):
        if self.task!='basket':return self.teacher.predict(observation_array(obs))
        raw=torch.as_tensor(obs,device=self.device,dtype=torch.float32)
        return self.model.act(raw).cpu().numpy()

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
            raw=rng.normal(size=(12,66)).astype(np.float32);raw[:,65]=np.arange(12)%2
            native=self.predict(raw)
            with torch.no_grad():
                x=self.model.normalize(torch.as_tensor(raw,device=self.device))
                h=self.model.actor[:-1](x)
                reference_prediction=torch.where(x[:,65:66]>.5,self.model.carry_head(h),self.model.actor[-1](h)).tanh().cpu().numpy()
            inference=float(np.max(np.abs(native-reference_prediction)))
            reference=adapter.action_dict(native)
        if inference>1e-5:raise ValueError('Inference mismatch')
        command=adapter.canonical(native)
        for key,value in adapter.action_dict(command).items():np.testing.assert_allclose(value,reference[key],atol=1e-7)
        return dict(sha256=self.hash,inference_max_error=inference,observation_parity=True,physical_action_parity=True)
