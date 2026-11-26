import unittest,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from .environment import AutonomousTeacherEnv,ready_to_throw
from .policy import Policy
from .run import validate_initialization,package
ROOT=Path(__file__).resolve().parents[1]
class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config=json.loads((ROOT/'basket_mobile_bridge/configs/tennis.json').read_text())
        o=cls.config['options'];o['pose_bank_entries']=json.loads((ROOT/o.pop('pose_bank')).read_text())
    def test_current_weights_load(self):
        source=torch.load(ROOT/'basket_mobile_curriculum/runs/tennis_mobile_curriculum_seed123/teacher_last.pth',map_location='cpu',weights_only=False)
        validate_initialization(source,self.config);m=Policy();m.load_state_dict(source['model'],strict=True)
        self.assertTrue(all(p.requires_grad for p in m.parameters()))
        self.assertEqual(package(m,self.config,0)['schema'],'basket_mobile_bridge_v1')
    def test_handoff_without_physical_reset(self):
        env=AutonomousTeacherEnv(dict(self.config['options'],evaluation_task='mobile'))
        try:
            obs,_=env.reset(seed=123);self.assertEqual(obs[-1],1.)
            qpos=env.Dcmm.data.qpos.copy();qvel=env.Dcmm.data.qvel.copy();targets=env.Dcmm.target_arm_qpos.copy()
            info=dict(curriculum_task='mobile',reward_terms=dict(time=-.02,score=0.,failure=0.),carry_diagnostics=dict(slip_speed=0.,in_support_envelope=True),
                hand_contact=True,base_speed=0.,success=False,task_success=False,safe_basket_success=False)
            with patch('basket_mobile_curriculum.environment.AutonomousTeacherEnv.step',return_value=(obs,0.,False,False,info)),patch('basket_mobile_bridge.environment.ready_to_throw',return_value=True):
                for _ in range(3):result=env.step(np.zeros(20))
                self.assertTrue(result[4]['reached_throw_phase']);self.assertEqual(result[0][-1],0.)
                self.assertEqual(result[4]['reward_terms']['handoff'],5.)
                result=env.step(np.zeros(20));self.assertEqual(result[4]['reward_terms']['handoff'],0.)
            np.testing.assert_array_equal(qpos,env.Dcmm.data.qpos);np.testing.assert_array_equal(qvel,env.Dcmm.data.qvel)
            np.testing.assert_array_equal(targets,env.Dcmm.target_arm_qpos)
        finally:env.close()
    def test_premature_release(self):
        env=AutonomousTeacherEnv(dict(self.config['options'],evaluation_task='mobile'))
        try:
            obs,_=env.reset(seed=123);env._released_once=True
            info=dict(curriculum_task='mobile',reward_terms=dict(score=100.,release=8.,ballistic_progress=4.,failure=0.),success=True,task_success=True,safe_basket_success=True)
            with patch('basket_mobile_curriculum.environment.AutonomousTeacherEnv.step',return_value=(obs,0.,False,False,info)):
                result=env.step(np.zeros(20));self.assertTrue(result[2]);self.assertFalse(result[4]['task_success'])
                self.assertEqual(result[4]['reward_terms']['score'],0.);self.assertEqual(result[4]['terminated_reason'],'premature_bridge_release')
        finally:env.close()
    def test_ready_region(self):
        info=dict(hand_contact=True,base_speed=0.,carry_diagnostics=dict(slip_speed=0.,in_support_envelope=True))
        self.assertTrue(ready_to_throw(info,1.,0.));self.assertFalse(ready_to_throw(info,1.8,0.))
        self.assertFalse(ready_to_throw(info,1.,.2));info['hand_contact']=False;self.assertFalse(ready_to_throw(info,1.,0.))
    def test_actual_step(self):
        for task in ('mobile','fixed','carry'):
            env=AutonomousTeacherEnv(dict(self.config['options'],evaluation_task=task,episode_seconds=.04))
            try:
                obs,_=env.reset(seed=123);result=env.step(np.zeros(20));self.assertEqual(result[0].shape,(66,))
                self.assertTrue(np.isfinite(result[1]));self.assertTrue(result[2] or result[3])
                self.assertEqual(obs[-1],float(task!='fixed'))
            finally:env.close()
if __name__=='__main__':unittest.main()
