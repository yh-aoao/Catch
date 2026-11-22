"""Physical check that the student, not the phase controller, controls all channels."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import json
from pathlib import Path
import unittest
import numpy as np
from .factories import AutonomousBasketEnv
from .run import env_config

class AutonomousControls(unittest.TestCase):
    def test_navigation_does_not_freeze_arm_or_fingers(self):
        c=json.loads((Path(__file__).parent/'configs/catch.json').read_text())
        kw=env_config(c,'basket')['env_kwargs']
        env=AutonomousBasketEnv(**dict(kw,task='Catching',object_motion='throw_basket',viewer=False,render_mode=None))
        self.addCleanup(env.close);env.render=lambda:np.empty(0,np.float32)
        env.reset(seed=123)
        self.assertFalse(env.basket_mobile_state['unlocked'])
        held=env.basket_carry_arm.copy();hand=env.Dcmm.target_hand_qpos.copy()
        env.step(dict(base=np.array([.06,0.]),arm=np.full(6,.005),hand=np.full(12,.015)))
        self.assertGreater(float(np.max(np.abs(env.Dcmm.target_arm_qpos-held))),.001)
        self.assertGreater(float(np.max(np.abs(env.Dcmm.target_hand_qpos-hand))),.001)
        self.assertGreater(float(np.linalg.norm(env.Dcmm.target_base_vel[:2])),0.)
        env.basket_mobile_state.update(unlocked=True,ready_time=float(env.Dcmm.data.time-env.start_time))
        # Base command must remain available even when the scoring state says 'throw'.
        env.step(dict(base=np.array([.06,0.]),arm=np.zeros(6),hand=np.zeros(12)))
        self.assertGreater(float(np.linalg.norm(env.Dcmm.target_base_vel[:2])),0.)

if __name__=='__main__':unittest.main()
