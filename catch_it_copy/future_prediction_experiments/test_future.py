"""CPU synthetic regression, probabilistic-head, masking and resume tests."""
import copy
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from visual_module_experiments import test_modules as old
from visual_module_experiments.test_modules import FakePool
from distillation_visual.tests.test_visual import FakeTeacher
from future_prediction_experiments.core import EpisodeReplay, StableStudent, UncertainStudent
from future_prediction_experiments.training import learning_rate


def config(variant,stage='catch'):
    c = old.Tests().configuration('future_'+variant,stage)
    c.update(eval_start=1,evaluation_seed=20261110,diagnostics_every=1)
    c['lr_schedule'] = dict(kind='cosine',start=1,end=3,minimum=3e-5) if variant=='stable' else dict(kind='constant')
    if variant=='uncertain':
        c['model']['future_prediction'] = dict(horizons=[1,3,5,8],target_scale_m=.1,logvar_bounds=[-6.,4.])
    c['max_episode_steps'] = max(c['max_episode_steps'],16)
    c['capacity_frames_per_task'] = max(c['capacity_frames_per_task'],32)
    c['skip_teacher_baseline'] = True
    return c


def inputs(student,n=6):
    return (torch.rand(1,n,*student.image_shape),torch.rand(1,n,student.proprio_dim),torch.rand(1,n,student.action_dim))


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_original_future_exact_parity(self):
        from distillation_aux_future.core import Student
        c=config('stable')
        torch.manual_seed(23);a=Student(c['stage'],c['vision'],c['model'],c['auxiliary'])
        torch.manual_seed(23);b=StableStudent(c['stage'],c['vision'],c['model'],c['auxiliary'])
        self.assertEqual(list(a.state_dict()),list(b.state_dict()))
        for key,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[key],rtol=0,atol=0)
        x=inputs(a)
        torch.testing.assert_close(a(*x)[0],b(*x)[0],rtol=0,atol=0)

    def test_replay_contact_mask_offsets_and_resume(self):
        from visual_module_experiments.core import EpisodeReplay as Original
        rows=[]
        for j in range(12):
            rows.append(dict(depth=np.zeros((2,64,64),np.float16),proprio=np.zeros(24,np.float32),
                previous=np.zeros(20,np.float32),target=np.zeros(20,np.float32),object_target=np.ones(6,np.float32),
                object_valid=np.ones(6,np.float32),world_position=np.array([j,0,abs(j-3)],np.float32),
                visible=np.zeros(2,np.float32),uv=np.zeros((2,2),np.float32),contact_after=np.float32(j==6)))
        a=Original(['throw'],50);b=EpisodeReplay(['throw'],50)
        a.add('throw',rows);b.add('throw',rows)
        for key in ('future_target','future_valid'):
            torch.testing.assert_close(a.data['throw'][0][key],b.data['throw'][0][key])
        r=EpisodeReplay(['throw'],50,[1,3,5,8]);r.add('throw',rows)
        e=r.data['throw'][0]
        self.assertEqual(e['future_target'][0,3,0],8)
        self.assertEqual(e['future_valid'][0,3],0) # Contact at 6 precedes endpoint 8.
        self.assertEqual(e['future_valid'][1,2],1) # Endpoint 6 before its outgoing contact interval.
        self.assertEqual(e['future_valid'][2,2],0)
        self.assertEqual(e['forecast_phase'][4],1)
        self.assertEqual(e['forecast_phase'][7],3)
        rr=EpisodeReplay(['throw'],50,[1,3,5,8]);rr.load_state_dict(r.state_dict())
        torch.testing.assert_close(rr.data['throw'][0]['future_target'],e['future_target'])
        self.assertEqual(rr.sample(1,'cpu')['future_target'].shape,(1,12,4,3))

    def test_uncertainty_gradient_sequence_and_padding(self):
        c=config('uncertain');run=importlib.import_module('distillation_aux_future_uncertain.run')
        s=run.make_student(c);x=inputs(s)
        action,_,_=s(*x);hidden=None;parts=[]
        for j in range(6):
            a,hidden,_=s(*(v[:,j:j+1] for v in x),hidden);parts.append(a)
        torch.testing.assert_close(action,torch.cat(parts,1),atol=1e-6,rtol=1e-5)
        # Imitation cannot directly tune confidence parameters.
        s.zero_grad();s(*x)[0].sum().backward()
        self.assertTrue(all(p.grad is None for p in s.uncertainty.parameters()))
        self.assertGreater(sum(p.grad.abs().sum().item() for p in s.forecast.parameters()),0)
        s.zero_grad();s.forward_aux(*x)
        batch=dict(valid=torch.tensor([[1.,1,1,1,0,0]]),future_target=torch.ones(1,6,4,3)*.03,
                   future_valid=torch.ones(1,6,4))
        loss=s.extra_loss(batch,0,6,torch.tensor([4.]))
        changed=copy.deepcopy(batch);changed['future_target'][:,4:]=float('nan')
        torch.testing.assert_close(loss,s.extra_loss(changed,0,6,torch.tensor([4.])))
        loss.backward()
        self.assertGreater(s.uncertainty[-1].weight.grad.abs().sum().item(),0)
        logvar=s.extra['logvar'];self.assertTrue(bool(((logvar>=-6)&(logvar<=4)).all()))
        # Empty supervision yields zero and no uncertainty gradient contribution.
        s.zero_grad();s.forward_aux(*x);batch['future_valid'].zero_()
        zero=s.extra_loss(batch,0,6,torch.tensor([4.]));self.assertEqual(zero.item(),0)
        zero.backward()
        self.assertEqual(sum(p.grad.abs().sum().item() for p in s.uncertainty.parameters()),0)

    def test_lr_boundaries(self):
        c=dict(learning_rate=3e-4,lr_schedule=dict(kind='cosine',start=5000,end=10000,minimum=3e-5))
        self.assertAlmostEqual(learning_rate(c,1),3e-4)
        self.assertAlmostEqual(learning_rate(c,5000),3e-4)
        self.assertAlmostEqual(learning_rate(c,7500),1.65e-4)
        self.assertAlmostEqual(learning_rate(c,10000),3e-5)
        self.assertAlmostEqual(learning_rate(c,12000),3e-5)

    def test_probabilistic_loss_chunk_equivalence_and_label_free_inference(self):
        run=importlib.import_module('distillation_aux_future_uncertain.run')
        c=config('uncertain');s=run.make_student(c);x=inputs(s)
        s.forward_aux(*x)
        full_extra=dict(s.extra)
        batch=dict(valid=torch.tensor([[1.,1,1,1,0,0]]),future_target=torch.rand(1,6,4,3),
                   future_valid=torch.ones(1,6,4))
        whole=s.extra_loss(batch,0,6,torch.tensor([4.]))
        parts=[]
        for begin,end in ((0,2),(2,4),(4,6)):
            s.extra={k:v[:,begin:end] for k,v in full_extra.items()}
            parts.append(s.extra_loss(batch,begin,end,torch.tensor([4.])))
        torch.testing.assert_close(whole,sum(parts))
        pool=FakePool('throw',c,1,0);previous=np.zeros((1,s.action_dim),np.float32)
        before=run.student_step(s,pool.obs,previous,None,'catch')[0]
        for k in ('world_position','object_target','uv','visible'):
            pool.obs[k][:]=np.nan
        pool.obs['teacher_obs']['object']['pos3d'][:]=np.nan
        np.testing.assert_array_equal(before,run.student_step(s,pool.obs,previous,None,'catch')[0])

    def test_default_configs_and_evaluation_schedule(self):
        root=Path(__file__).resolve().parents[1]
        for variant in ('stable','uncertain'):
            run=importlib.import_module('distillation_aux_future_'+variant+'.run')
            for stage in ('catch','track'):
                c=json.loads((root/('distillation_aux_future_'+variant)/'configs'/ (stage+'.json')).read_text())
                run.validate(c)
                self.assertEqual([i for i in range(1,10001) if run.should_evaluate(i,c)],
                                 [5000,6000,7000,8000,9000,10000])

    def test_train_save_eval_resume_both_variants_and_stages(self):
        for variant in ('stable','uncertain'):
            for stage in ('catch','track'):
                run=importlib.import_module('distillation_aux_future_'+variant+'.run');c=config(variant,stage)
                adapters=run.validate(c);teachers={t:FakeTeacher(20 if stage=='catch' else 8) for t in c['tasks']}
                with tempfile.TemporaryDirectory() as temp,patch.object(run,'Pool',FakePool),patch.object(run,'provenance',return_value={'test':'synthetic'}):
                    out=Path(temp);run.train(c,teachers,adapters,out)
                    saved=torch.load(out/'resume_last.pth',weights_only=True)
                    self.assertEqual(saved['schema'],'distillation_aux_future_'+variant+'_v1')
                    self.assertEqual(saved['iteration'],2)
                    self.assertTrue((out/'student_000002.pth').exists())
                    s=run.make_student(c);s.load_state_dict(saved['student'],strict=True)
                    self.assertTrue(all(r['success_rate']==1 for r in run.evaluate(c,adapters,student=s).values()))
                    c['iterations']=3;run.train(c,teachers,adapters,out,out/'resume_last.pth')
                    saved=torch.load(out/'resume_last.pth',weights_only=True)
                    self.assertEqual(saved['iteration'],3)
                    self.assertAlmostEqual(saved['optimizer']['param_groups'][0]['lr'],learning_rate(c,3))
                    logs=[json.loads(l) for l in (out/'metrics.jsonl').read_text().splitlines()]
                    self.assertTrue(any('diagnostic/bounce/frames' in r for r in logs))


if __name__=='__main__':
    unittest.main()
