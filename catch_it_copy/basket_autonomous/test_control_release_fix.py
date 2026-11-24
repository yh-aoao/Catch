"""Regression checks for the two fixes; no optimizer updates."""
import copy,json,unittest
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from .environment import AutonomousTeacherEnv

class ReleaseTests(unittest.TestCase):
    def tracker(self):
        s=NS(Dcmm=NS(data=NS(time=0.,qvel=np.zeros(42)),model=NS(opt=NS(gravity=np.array([0.,0.,-9.81])))),
            basket_center=np.array([0.,1.,.55]),basket_cfg=NS(basket_tilt_deg=0.),
            _low_speed_time=.2,_no_hand_seconds=0.,_last_touch=None,_release=None,
            _released_once=False,_release_reward=0.,_ever_intentional_launch=False,
            _hand_clearance=lambda hands:.02)
        s.Dcmm.data.qvel[36:39]=[0.,2.,1.]
        return s
    def tick(self,s,time,touch,position):
        s.Dcmm.data.time=time
        AutonomousTeacherEnv._update_confirmed_release(s,touch,np.array(position),0.,.02,{1})
    def test_tiny_gap_recontact_does_not_pay_or_latch(self):
        s=self.tracker();self.tick(s,0,True,[0,0,.7])
        for i in range(1,8):self.tick(s,i*.02,False,[0,.005,.7])
        self.assertIsNotNone(s._release);self.assertFalse(s._released_once);self.assertEqual(s._release_reward,0.)
        self.tick(s,.16,True,[0,0,.7]);self.assertIsNone(s._release)
        self.assertFalse(s._released_once)
    def test_confirmed_flight_pays_once(self):
        s=self.tracker();self.tick(s,0,True,[0,0,.7])
        for i in range(1,7):self.tick(s,i*.02,False,[0,i*.04,.7+i*.01])
        self.assertTrue(s._released_once);self.assertTrue(s._release['confirmed'])
        self.assertTrue(s._ever_intentional_launch);self.assertGreater(s._release_reward,0.)
        s._release_reward=0.;self.tick(s,.14,False,[0,.28,.77]);self.assertEqual(s._release_reward,0.)
    def test_near_hand_cannot_confirm_even_after_travel(self):
        s=self.tracker();s._hand_clearance=lambda hands:.001
        self.tick(s,0,True,[0,0,.7])
        for i in range(1,7):self.tick(s,i*.02,False,[0,i*.04,.7])
        self.assertFalse(s._released_once);self.assertEqual(s._release_reward,0.)
    def test_drop_does_not_get_positive_launch_reward(self):
        s=self.tracker();s.Dcmm.data.qvel[36:39]=[0,0,-.5]
        self.tick(s,0,True,[0,0,.7])
        for i in range(1,7):self.tick(s,i*.02,False,[0,0,.7-i*.04])
        self.assertTrue(s._released_once);self.assertFalse(s._ever_intentional_launch)
        self.assertEqual(s._release_reward,-2.)

class ControlTests(unittest.TestCase):
    def test_real_zero_velocity_retains_target_and_motion_changes_it(self):
        c=json.loads((Path(__file__).parent/'configs/fixed_carry_v2.json').read_text())
        env=AutonomousTeacherEnv(c['options'])
        try:
            np.random.seed(500123);env.reset(seed=500123)
            initial=env.Dcmm.target_arm_qpos.copy()
            for _ in range(25):
                _,r,done,trunc,info=env.step(np.zeros(20,np.float32))
                self.assertFalse(done or trunc)
            np.testing.assert_allclose(env.Dcmm.target_arm_qpos,initial,atol=1e-10)
            self.assertTrue(env.touching());self.assertFalse(env._released_once)
            previous=env.Dcmm.target_arm_qpos.copy()
            command=np.zeros(20,np.float32);command[2:8]=.1
            env.step(command)
            self.assertGreater(np.linalg.norm(env.Dcmm.target_arm_qpos-previous),0.)
            self.assertEqual(float(env.Dcmm.data.ctrl[-1]),0.)
        finally:env.close()
    def test_release_surface_clearance_api(self):
        c=json.loads((Path(__file__).parent/'configs/fixed_carry_v2.json').read_text())
        env=AutonomousTeacherEnv(c['options'])
        try:
            env.reset(seed=123)
            from gym_dcmm.utils.roll_rewards import hand_collision_ids
            hands=set(hand_collision_ids(env.Dcmm.model,env.hand_start_id).tolist())
            self.assertTrue(np.isfinite(env._hand_clearance(hands)))
        finally:env.close()

if __name__=='__main__':unittest.main()
