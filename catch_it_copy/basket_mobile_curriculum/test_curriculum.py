import unittest,json,copy
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch
from .policy import Policy
from .environment import AutonomousTeacherEnv,LEVELS
from .run import validate_initialization,package,evaluation_options
from .pool import Pool
ROOT=Path(__file__).resolve().parents[1]
class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config=json.loads((ROOT/'basket_mobile_curriculum/configs/tennis.json').read_text())
        o=cls.config['options'];o['pose_bank_entries']=json.loads((ROOT/o.pop('pose_bank')).read_text())
        cls.saved=torch.load(ROOT/'basket_autonomous/runs/tennis_hold_v2_fixed_fix_seed123/teacher_best.pth',map_location='cpu',weights_only=False)
    def test_weight_conversion_and_exploration(self):
        from basket_autonomous.policy import Policy as Old
        m=Policy();m.load_fixed(self.saved['model']);old=Old();old.load_state_dict(self.saved['model'])
        x=torch.randn(4,65);throw=torch.cat([x,torch.zeros(4,1)],1);carry=torch.cat([x,torch.ones(4,1)],1)
        self.assertTrue(torch.allclose(m.act(throw),old.act(x),atol=1e-6))
        self.assertTrue(torch.equal(m.act(carry),torch.zeros(4,20)))
        self.assertTrue(torch.allclose(m.distribution(carry).scale[:,2:],m.distribution(throw).scale[:,2:]*.05))
        self.assertTrue(all(p.requires_grad for p in m.parameters()))
        validate_initialization(self.saved,self.config)
    def test_tasks_and_terminal_rewards(self):
        for task in ('carry','fixed','far_mobile'):
            env=AutonomousTeacherEnv(dict(self.config['options'],evaluation_task=task,episode_seconds=.04))
            try:
                obs,_=env.reset(seed=123);self.assertEqual(obs.shape,(66,));self.assertEqual(obs[-1],float(task=='carry'))
                self.assertEqual(env.fixed_episode,task=='fixed')
                _,reward,term,trunc,info=env.step(np.zeros(20));self.assertTrue(term or trunc);self.assertTrue(np.isfinite(reward))
                if task=='carry':
                    self.assertEqual(info['reward_terms']['failure'],-20.)
                    self.assertNotIn('release_distance',info['reward_terms']);self.assertFalse(info['safe_basket_success'])
                if task=='far_mobile':self.assertGreaterEqual(env.basket_center[1],1.8)
            finally:env.close()
    def test_carry_success_and_release_failure(self):
        env=AutonomousTeacherEnv(dict(self.config['options'],evaluation_task='carry'))
        try:
            env.reset(seed=123)
            info=dict(carry_diagnostics=dict(slip_speed=0.,in_support_envelope=True),hand_contact=True,
                base_speed=0.,reward_terms=dict(time=-.02,score=100.,failure=0.,release=8.),safe_basket_success=True)
            env._released_once=False;env.steps=1
            with patch('basket_autonomous.environment.AutonomousTeacherEnv.step',return_value=(env.observation(),0.,False,False,info)),patch.object(env,'_navigation_distance',return_value=0.):
                for _ in range(8):result=env.step(np.zeros(20))
                self.assertTrue(result[2]);self.assertTrue(result[4]['carry_success'])
                self.assertFalse(result[4]['safe_basket_success']);self.assertEqual(result[4]['reward_terms']['carry_success'],30.)
                env._released_once=True
                result=env.step(np.zeros(20));self.assertTrue(result[2]);self.assertFalse(result[4]['task_success'])
                self.assertEqual(result[4]['terminated_reason'],'carry_release')
        finally:env.close()
    def test_promotion_gate(self):
        from .run import gate_streak_after
        self.assertEqual(gate_streak_after(0,0,.8),1)
        self.assertEqual(gate_streak_after(0,1,.8),2)
        self.assertEqual(gate_streak_after(0,1,.79),0)
        self.assertEqual(gate_streak_after(2,1,.7),2)
    def test_spawn_pool_and_checkpoint(self):
        pool=Pool(2,dict(self.config['options'],evaluation_task='carry',episode_seconds=.04),123)
        try:
            self.assertEqual(pool.obs.shape,(2,66));r=pool.step(np.zeros((2,20)))
            self.assertTrue(np.isfinite(r[1]).all());self.assertEqual(r[-1].shape,(2,66))
            self.assertTrue(all(i['curriculum_task']=='carry' for i in r[4]))
        finally:pool.close()
        model=Policy();saved=package(model,self.config,0);other=Policy();other.load_state_dict(saved['model'])
        self.assertEqual(saved['schema'],'basket_mobile_curriculum_v1')
        self.assertEqual(evaluation_options(self.config)['evaluation_task'],'far_mobile')
if __name__=='__main__':unittest.main()
