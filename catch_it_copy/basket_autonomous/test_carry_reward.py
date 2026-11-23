import copy,json,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import mujoco
import torch
from .carry_reward import carry_terms,navigation_progress,region_distance,purposeful_launch
from .environment import AutonomousTeacherEnv
from .policy import Policy
from .run import package,validate_initialization

class RewardTests(unittest.TestCase):
    def test_stability_and_no_idle_farming(self):
        stable,d=carry_terms([0,.05,.07],[0,0,0],True,False,0,0,.04)
        sliding,_=carry_terms([0,.05,.07],[0,.3,0],True,False,0,0,.04)
        self.assertGreater(sum(stable.values()),sum(sliding.values()))
        self.assertLess(sum(stable.values())-.02,0.)
        # Fast forward swing and confirmed release disable carrying costs/rewards.
        for released,speed in [(True,0),(False,1.)]:
            r,_=carry_terms([0,.20,.07],[0,2,0],True,released,speed,speed,.04)
            self.assertEqual(sum(abs(v) for v in r.values()),0.)
    def test_edge_direction(self):
        outward,_=carry_terms([0,.198,.07],[0,.1,0],True,False,0,0,.04)
        inward,_=carry_terms([0,.198,.07],[0,-.1,0],True,False,0,0,.04)
        self.assertLess(outward['carry_edge'],inward['carry_edge'])
    def test_navigation_range_and_no_loops(self):
        for p in ([0,.9,0],[.05,1.1,0]):self.assertEqual(region_distance(p,[-.1,.1],[.85,1.15]),0.)
        best=2.;total=0.
        for distance in [1.5,.7,1.5,.7,.5,1.,.5,0.]:
            best,r=navigation_progress(best,distance,True);total+=r
        self.assertAlmostEqual(total,4.)
        self.assertEqual(navigation_progress(.5,0.,False),(.5,0.))
    def test_launch_classification_is_not_hit_or_stop_test(self):
        self.assertTrue(purposeful_launch([0,0,.7],[0,1.,.4],[0,1.,.55],.8))
        self.assertFalse(purposeful_launch([0,0,.7],[0,.1,-.1],[0,1.,.55],None))
        self.assertFalse(purposeful_launch([0,0,.7],[0,-2.,1.],[0,1.,.55],.1))

class PhysicalRewardTests(unittest.TestCase):
    def config(self,stage):
        c=json.loads((Path(__file__).parent/'configs'/f'{stage}_carry_v2.json').read_text())
        bank=c['options'].pop('pose_bank')
        c['options']['pose_bank_entries']=json.loads(Path(bank).read_text())
        return c
    def test_new_stage_initialization_and_protocol(self):
        source=package(Policy(),self.config('fixed'),0)
        validate_initialization(source,self.config('mobile'))
        old=copy.deepcopy(source);old['config']['options'].pop('reward_version')
        with self.assertRaises(ValueError):validate_initialization(old,self.config('mobile'))
    def test_reward_version_does_not_change_physics(self):
        envs=[]
        try:
            states=[]
            for version in ('autonomous_v1','carry_navigation_v2'):
                c=self.config('mobile');c['options']['fixed_replay_probability']=0.;c['options']['reward_version']=version
                np.random.seed(17);env=AutonomousTeacherEnv(c['options']);envs.append(env)
                np.random.seed(124);env.reset(seed=124)
                trajectory=[]
                for _ in range(5):
                    env.step(np.full(20,.1,np.float32));trajectory.append(env.Dcmm.data.qpos.copy())
                states.append(trajectory)
            np.testing.assert_allclose(states[0],states[1],atol=1e-10,rtol=0.)
        finally:
            for env in envs:env.close()

    def test_physics_metrics_no_ball_actuation(self):
        torch.set_num_threads(1)
        for stage in ('fixed','mobile'):
            c=self.config(stage);c['options']['fixed_replay_probability']=0.
            env=AutonomousTeacherEnv(c['options'])
            try:
                env.reset(seed=123)
                # Stable carrier translation must not create relative slip or swing intent.
                d=env.Dcmm.data;d.qvel[:]=0.;mujoco.mj_forward(env.Dcmm.model,d)
                before=env._carry_measurements()
                d.qvel[:2]=[.2,.1];d.qvel[36:39]=[.2,.1,0.];mujoco.mj_forward(env.Dcmm.model,d)
                after=env._carry_measurements()
                np.testing.assert_allclose(before[1],after[1],atol=1e-7)
                np.testing.assert_allclose(before[2:],after[2:],atol=1e-7)
                env.reset(seed=124)
                with patch.object(env.Dcmm,'set_throw_pos_vel',side_effect=AssertionError('teleport')),patch.object(env,'_basket_parking_state',side_effect=AssertionError('waypoint')):
                    obs,reward,done,trunc,info=env.step(np.full(20,.1,np.float32))
                self.assertAlmostEqual(reward,sum(info['reward_terms'].values()))
                self.assertIn('carry_stability',info['reward_terms']);self.assertIn('carry_navigation',info['reward_terms'])
                if stage=='fixed':self.assertEqual(info['reward_terms']['carry_navigation'],0.)
                self.assertTrue(np.isfinite(info['carry_slip_mean']))
                self.assertEqual(float(env.Dcmm.data.ctrl[-1]),0.)
                self.assertTrue(np.all(env.Dcmm.data.qfrc_applied[36:42]==0.))
                self.assertTrue(np.all(env.Dcmm.data.xfrc_applied[env.Dcmm.model.body('object').id]==0.))
            finally:env.close()
if __name__=='__main__':unittest.main()
