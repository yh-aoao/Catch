"""CPU synthetic contract tests; no claim of MuJoCo/GPU validation."""
import copy
import importlib
import tempfile
import unittest
import types
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from distillation_visual_aux.tests.test_auxiliary import config, FakePool as BasePool
from distillation_visual.tests.test_visual import FakeTeacher
from visual_module_experiments.scaffold import VARIANTS


class FakePool(BasePool):
    def packet(self):
        p = super().packet()
        p.update(world_position=np.repeat(self.steps[:,None],3,axis=1).astype(np.float32),
                 visible=np.ones((self.count,2),np.float32),uv=np.zeros((self.count,2,2),np.float32))
        return p

    def step(self, actions):
        p,r,d,info = super().step(actions)
        for row in info: row['module_hand_contact'] = False
        return p,r,d,info


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def configuration(self, variant, stage='catch'):
        c = config(stage)
        c['experiment'] = {'variant':variant}
        c['auxiliary']['velocity_weight'] = 0
        return c

    def test_sequence_equivalence_gradients_and_inference(self):
        for variant,name in VARIANTS.items():
            for stage in ('track','catch'):
                run=importlib.import_module(name+'.run'); c=self.configuration(variant,stage)
                s=run.make_student(c)
                inputs=(torch.rand(1,6,2,64,64),torch.rand(1,6,s.proprio_dim),torch.rand(1,6,s.action_dim))
                a,h,_=s(*inputs)
                hh=None; parts=[]
                for j in range(6):
                    aa,hh,_=s(*(v[:,j:j+1] for v in inputs),hh);parts.append(aa)
                torch.testing.assert_close(a,torch.cat(parts,1),atol=1e-6,rtol=1e-5)
                batch=dict(depth=inputs[0],proprio=inputs[1],previous=inputs[2],target=torch.zeros_like(a),
                           valid=torch.tensor([[1.,1,1,1,1,0]]),object_target=torch.ones(1,6,6),
                           object_valid=torch.ones(1,6,6),future_target=torch.ones(1,6,3,3),future_valid=torch.ones(1,6,3),
                           visible=torch.ones(1,6,2),uv=torch.zeros(1,6,2,2),contact_target=torch.ones(1,6,3),contact_valid=torch.ones(1,6,3))
                optimizer=torch.optim.Adam(s.parameters(),lr=.001)
                losses=run.optimize(s,optimizer,batch,c)
                self.assertTrue(all(np.isfinite(v) for v in losses.values()))
                for e in s.encoders: self.assertGreater(e[0][0].weight.grad.abs().sum().item(),0)
                module = {'future':'forecast','belief':'confidence','spatial':'heatmaps',
                          'contact':'phase','camera_gate':'gate'}[variant]
                self.assertGreater(sum(p.grad.abs().sum().item() for p in getattr(s,module).parameters() if p.grad is not None),0)
                # Labels never enter student_step, even if poisoned.
                pool=FakePool('throw',c,1,0);prev=np.zeros((1,s.action_dim),np.float32)
                before=run.student_step(s,pool.obs,prev,None,stage)[0]
                for k in ('world_position','visible','uv','object_target'):pool.obs[k][:]=np.nan
                pool.obs['teacher_obs']['object']['pos3d'][:]=np.nan
                np.testing.assert_array_equal(before,run.student_step(s,pool.obs,prev,None,stage)[0])

    def test_replay_future_and_censored_contact(self):
        from visual_module_experiments.core import EpisodeReplay
        r=EpisodeReplay(['throw'],50)
        rows=[]
        for j in range(7):
            rows.append(dict(depth=np.zeros((2,64,64),np.float16),proprio=np.zeros(24,np.float32),previous=np.zeros(20,np.float32),
                             target=np.zeros(20,np.float32),object_target=np.zeros(6,np.float32),object_valid=np.ones(6,np.float32),
                             world_position=np.array([j,0,0],np.float32),visible=np.ones(2,np.float32),uv=np.zeros((2,2),np.float32),contact_after=np.float32(j==3)))
        r.add('throw',rows);e=r.data['throw'][0]
        self.assertEqual(e['future_target'][0,1,0],3)
        self.assertEqual(e['future_valid'][0,2],0)
        self.assertEqual(e['contact_target'][1,0],1)
        self.assertEqual(e['contact_valid'][-1,0],0)
        self.assertEqual(e['future_valid'][-1].sum(),0)
        self.assertEqual(e['contact_target'][4,1],1)
        rr=EpisodeReplay(['throw'],50);rr.load_state_dict(r.state_dict())
        torch.testing.assert_close(rr.sample(1,'cpu')['future_target'][0],e['future_target'])

    def test_segmentation_labels_and_restore_depth(self):
        from visual_module_experiments.labels import labels
        class Renderer:
            mode='depth'
            camera=None
            def disable_depth_rendering(self): self.mode='off'
            def enable_depth_rendering(self): self.mode='depth'
            def enable_segmentation_rendering(self): self.mode='segmentation'
            def disable_segmentation_rendering(self): self.mode='off'
            def update_scene(self,data,camera): self.camera=camera
            def render(self):
                self.assert_mode = self.mode
                x=np.full((5,5,2),-1,dtype=int)
                if self.camera=='top':x[2,3]=[7,5]
                return x
        renderer=Renderer()
        env=types.SimpleNamespace(Dcmm=types.SimpleNamespace(
            model=types.SimpleNamespace(geom=lambda name:types.SimpleNamespace(id=7)),
            data=types.SimpleNamespace(geom_xpos=np.zeros((8,3)))))
        sensor=types.SimpleNamespace(renderer=renderer,mujoco=types.SimpleNamespace(mjtObj=types.SimpleNamespace(mjOBJ_GEOM=5)))
        target=labels(env,sensor,[{'name':'top'},{'name':'wrist'}],True)
        np.testing.assert_array_equal(target['visible'],[1,0])
        np.testing.assert_allclose(target['uv'][0],[.5,0])
        self.assertEqual(renderer.mode,'depth')
        with patch.object(renderer,'render',side_effect=RuntimeError('render failed')):
            with self.assertRaises(RuntimeError):labels(env,sensor,[{'name':'top'}],True)
        self.assertEqual(renderer.mode,'depth')

    def test_extra_loss_padding_mask(self):
        for variant,name in VARIANTS.items():
            run=importlib.import_module(name+'.run');c=self.configuration(variant)
            s=run.make_student(c)
            inputs=(torch.rand(1,4,2,64,64),torch.rand(1,4,24),torch.rand(1,4,20))
            s.forward_aux(*inputs)
            batch=dict(valid=torch.tensor([[1.,1,0,0]]),future_target=torch.ones(1,4,3,3),future_valid=torch.ones(1,4,3),
                       visible=torch.ones(1,4,2),uv=torch.zeros(1,4,2,2),contact_target=torch.ones(1,4,3),contact_valid=torch.ones(1,4,3))
            loss=s.extra_loss(batch,0,4,torch.tensor([2.]))
            changed=copy.deepcopy(batch)
            changed['future_target'][:,2:]=100
            changed['uv'][:,2:]=10
            changed['visible'][:,2:]=0
            changed['contact_target'][:,2:]=0
            torch.testing.assert_close(loss,s.extra_loss(changed,0,4,torch.tensor([2.])))

    def test_train_save_eval_resume_all_variants_stages(self):
        for variant,name in VARIANTS.items():
            for stage in ('catch','track'):
                run=importlib.import_module(name+'.run');c=self.configuration(variant,stage)
                adapters=run.validate(c);teachers={t:FakeTeacher(20 if stage=='catch' else 8) for t in c['tasks']}
                with tempfile.TemporaryDirectory() as temp, patch.object(run,'Pool',FakePool),patch.object(run,'provenance',return_value={'synthetic':'test'}):
                    out=Path(temp);run.train(c,teachers,adapters,out)
                    saved=torch.load(out/'resume_last.pth',weights_only=True)
                    self.assertEqual(saved['schema'],name+'_v1')
                    s=run.make_student(c);s.load_state_dict(saved['student'])
                    result=run.evaluate(c,adapters,student=s)
                    self.assertTrue(all(x['success_rate']==1 for x in result.values()))
                    c['iterations']=3;run.train(c,teachers,adapters,out,out/'resume_last.pth')
                    self.assertEqual(torch.load(out/'resume_last.pth',weights_only=True)['iteration'],3)


if __name__ == '__main__':
    unittest.main()
