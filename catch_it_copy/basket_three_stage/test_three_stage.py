"""Curriculum, transfer, sampling and physical reset checks; no training."""
import copy,json,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from .environment import AutonomousTeacherEnv
from basket_autonomous.environment import AutonomousTeacherEnv as BaseEnv
from .policy import Policy
from .run import package,validate_initialization,evaluation_options

ROOT=Path(__file__).resolve().parents[1]
def config(stage):
    c=json.loads((Path(__file__).parent/'configs'/stage).read_text())
    bank=c['options'].pop('pose_bank')
    c['options']['pose_bank_entries']=json.loads((ROOT/bank).read_text())
    return c

class PolicyTests(unittest.TestCase):
    def test_task_bit_and_gradients(self):
        torch.set_num_threads(1);model=Policy();x=torch.randn(8,66);x[:,65]=0.
        y=x.clone();y[:,65]=1.
        model.update_normalizer(x)
        self.assertTrue((model.normalize(y)[:,65]==1).all())
        torch.testing.assert_close(model.act(x),model.act(y))
        state=model.normalize(y);action,raw,logp,value=model.sample(state)
        loss=-model.log_probability(model.distribution(state),raw.detach()).mean()+value.square().mean()
        loss.backward();self.assertTrue(torch.isfinite(model.actor[0].weight.grad).all())
        self.assertGreater(float(model.actor[0].weight.grad[:,65].abs().sum()),0.)
        self.assertEqual(action.shape,(8,20))
    def test_stage_order_checkpoint_roundtrip(self):
        torch.set_num_threads(1);model=Policy()
        c1=config('stage1_hold.json');c2=config('stage2_throw.json');c3=config('stage3_mobile.json')
        first=package(model,c1,100);validate_initialization(first,c2)
        second=package(model,c2,100);validate_initialization(second,c3)
        other=Policy();other.load_state_dict(first['model'],strict=True)
        torch.testing.assert_close(model.act(torch.zeros(2,66)),other.act(torch.zeros(2,66)))
        self.assertTrue(all(p.requires_grad for p in other.parameters()))
        with self.assertRaises(ValueError):validate_initialization(first,c3)
        old=copy.deepcopy(first);old['schema']='basket_autonomous_teacher_v1'
        with self.assertRaises(ValueError):validate_initialization(old,c2)
        self.assertTrue(evaluation_options(c3)['evaluation'])

class PhysicsTests(unittest.TestCase):
    def test_source_distribution_and_nearby_poses(self):
        c=config('stage1_hold.json');env=AutonomousTeacherEnv(c['options']);groups=[];modes=[]
        try:
            for seed in range(1200,1220):
                np.random.seed(seed);obs,info=env.reset(seed=seed)
                self.assertEqual(obs.shape,(66,));self.assertEqual(obs[-1],0.)
                self.assertTrue(env.touching());self.assertTrue(info['fixed_episode'])
                source=info['reset_source_task'];groups.append(source);modes.append(info['reset_mode'])
                if source!='generic':
                    self.assertLessEqual(info['reset_sampling']['placement']['lift'],.050)
            self.assertGreaterEqual(sum(g!='generic' for g in groups),18)
            for task in ('throw','roll','bounce'):self.assertGreaterEqual(groups.count(task),2)
            print('reset source counts:',{t:groups.count(t) for t in set(groups)})
        finally:env.close()
    def test_task_reward_and_no_assistance(self):
        for name,task,bit,fixed in [('stage1_hold.json','hold',0.,True),('stage2_throw.json','fixed_throw',1.,True),('stage3_mobile.json','mobile_throw',1.,False)]:
            c=config(name);c['options']['evaluation']=True;env=AutonomousTeacherEnv(c['options'])
            try:
                obs,info=env.reset(seed=123)
                self.assertEqual(obs[-1],bit);self.assertEqual(info['episode_task'],task)
                self.assertEqual(info['fixed_episode'],fixed)
                command=np.full(20,.1,np.float32)
                with patch.object(env.Dcmm,'set_throw_pos_vel',side_effect=AssertionError('teleport')),patch.object(env,'_basket_parking_state',side_effect=AssertionError('waypoint')):
                    obs,r,done,trunc,info=env.step(command)
                self.assertAlmostEqual(r,sum(info['reward_terms'].values()))
                if task=='hold':self.assertNotIn('ballistic_progress',info['reward_terms'])
                else:self.assertIn('ballistic_progress',info['reward_terms'])
                self.assertEqual(float(env.Dcmm.data.ctrl[-1]),0.)
                self.assertTrue(np.all(env.Dcmm.data.qfrc_applied[36:42]==0.))
                if not fixed:self.assertGreater(float(np.linalg.norm(env.Dcmm.target_base_vel[:2])),0.)
            finally:env.close()
    def test_hold_success_has_no_basket_score(self):
        c=config('stage1_hold.json');env=AutonomousTeacherEnv(c['options'])
        try:
            env.reset(seed=123);env._required_hold=.08;env._released_once=False
            fake_info=dict(success=False,safe_basket_success=False,terminated_reason=None)
            with patch.object(BaseEnv,'step',return_value=(np.zeros(66,np.float32),100.,False,False,fake_info)),patch.object(env,'_carry_measurements',return_value=(np.array([0,.05,.07]),np.zeros(3),0.,0.)),patch.object(env,'touching',return_value=True):
                env.step(np.zeros(20,np.float32));_,r,done,trunc,info=env.step(np.zeros(20,np.float32))
            self.assertTrue(done);self.assertTrue(info['success']);self.assertTrue(info['hold_success'])
            self.assertFalse(info['safe_basket_success']);self.assertLess(r,21.)
        finally:env.close()

if __name__=='__main__':unittest.main()
