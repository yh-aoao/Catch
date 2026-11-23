import os
os.environ.setdefault('MUJOCO_GL','egl')
import json,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from .policy import Policy
from .environment import AutonomousTeacherEnv

class FullPolicyTests(unittest.TestCase):
    def test_full_action_logprob_and_gradients(self):
        torch.set_num_threads(1);model=Policy();obs=torch.randn(8,65)
        model.update_normalizer(obs);state=model.normalize(obs)
        action,raw,logp,value=model.sample(state)
        torch.testing.assert_close(model.log_probability(model.distribution(state),raw),logp)
        self.assertEqual(action.shape,(8,20));self.assertTrue((action.abs()<=1).all())
        loss=-model.log_probability(model.distribution(state),raw.detach()).mean()+value.square().mean()
        loss.backward();self.assertTrue(torch.isfinite(model.actor[-1].weight.grad).all())

class PhysicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bank=json.loads((Path(__file__).parent/'configs/caught_pose_bank.json').read_text())
        cls.env=AutonomousTeacherEnv(dict(pose_bank_entries=bank))
    @classmethod
    def tearDownClass(cls):cls.env.close()
    def test_random_supported_resets(self):
        states=[];sources=[]
        for seed in range(400,408):
            obs,info=self.env.reset(seed=seed)
            self.assertEqual(obs.shape,(65,));self.assertTrue(self.env.touching());self.assertTrue(np.isfinite(obs).all())
            self.assertLess(np.linalg.norm(self.env.Dcmm.data.qvel[36:39]),.35)
            states.append(self.env.Dcmm.data.qpos[15:37].copy());sources.append(info['reset_sampling']['anchor'])
        self.assertGreater(float(np.std(states,axis=0).max()),.1)
        print('Accepted random-start anchors:',sources)
    def test_no_phase_mask_or_ball_actuation_during_step(self):
        env=self.env;env.reset(seed=123)
        env.basket_mobile_state.update(unlocked=True,ready_time=0.)
        before_arm=env.Dcmm.target_arm_qpos.copy();before_hand=env.Dcmm.target_hand_qpos.copy()
        command=np.r_[.2,.1,np.full(6,.2),np.full(12,.1)].astype(np.float32)
        # Task logic is not allowed to call the old ball placer or old parking reference during policy steps.
        with patch.object(env.Dcmm,'set_throw_pos_vel',side_effect=AssertionError('ball teleport')),patch.object(env,'_basket_parking_state',side_effect=AssertionError('old waypoint')):
            obs,reward,terminated,truncated,info=env.step(command)
        self.assertGreater(float(np.linalg.norm(env.Dcmm.target_base_vel[:2])),0.)
        self.assertGreater(float(np.linalg.norm(env.Dcmm.target_arm_qpos-before_arm)),0.)
        self.assertGreater(float(np.linalg.norm(env.Dcmm.target_hand_qpos-before_hand)),0.)
        self.assertEqual(float(env.Dcmm.data.ctrl[-1]),0.)
        self.assertTrue(np.all(env.Dcmm.data.qfrc_applied[36:42]==0))
        self.assertTrue(np.all(env.Dcmm.data.xfrc_applied[env.Dcmm.model.body('object').id]==0))
        self.assertTrue(np.isfinite(reward));self.assertTrue(np.isfinite(obs).all())
        self.assertFalse(any(k in info for k in ('parking_target','phase','progress','remaining_time')))

if __name__=='__main__':unittest.main()
