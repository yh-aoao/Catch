"""Reward the palm-facing side, without forcing the thumb to copy three fingers."""
from pathlib import Path
import unittest

import mujoco
import numpy as np

from gym_dcmm.envs.bounce_grasp import MOVABLE, configuration, finger_posture_costs
from gym_dcmm.envs.bounce_physical_v2 import make_bounce_physical_v2


OPTIONS = dict(finger_backfold_weight=.15, finger_prepare_sync_weight=.05,
               finger_contact_weight=1., thumb_opposition_weight=.5)


class FingerPostureMathTests(unittest.TestCase):
    def test_thumb_is_independent_and_small_finger_differences_are_allowed(self):
        cfg = configuration(OPTIONS)
        q = np.array([.3, .2, .2]*3+[1.5, -1., 1.])
        q[3:6] += .2
        costs = finger_posture_costs(q, [.02, .02, .02], cfg)
        self.assertEqual(costs, dict(backfold=0., prepare_sync=0.))
        q[-3:] = [-1., 1.5, -1.]
        self.assertEqual(costs, finger_posture_costs(q, [.02, .02, .02], cfg))

    def test_backfold_and_large_one_finger_disagreement_are_bounded(self):
        cfg = configuration(OPTIONS)
        q = np.array([.3, .2, .2]*4)
        q[:3] = [-.3, -.5, -.3]
        costs = finger_posture_costs(q, [-.05, .02, .02], cfg)
        self.assertAlmostEqual(costs['backfold'], 1./3.)
        self.assertGreater(costs['prepare_sync'], 0.)
        for value in finger_posture_costs(q*100, [-100, -100, -100], cfg).values():
            self.assertLessEqual(value, 1.)


class FingerPostureRuntimeTests(unittest.TestCase):
    def make(self, ball='default', options=None):
        return make_bounce_physical_v2(task='Catching', object_motion='bounce',
            test_object=ball, viewer=False, render_mode=None, object_eval=False,
            bounce_launch='baseball_intercept_v1', bounce_grasp_version='controlled_hold_v1',
            bounce_grasp=OPTIONS if options is None else options,
            bounce_material_file=Path(__file__).resolve().parents[1]/'ycb_objects/calibration_data/balls.three_ball_training_v1.json')

    def test_local_tip_side_does_not_depend_on_arm_orientation(self):
        env = self.make()
        try:
            env.reset(seed=0)
            model, data = env.Dcmm.model, env.Dcmm.data
            data.qpos[21+MOVABLE] = np.tile([.15, .1, .1], 4)
            mujoco.mj_forward(model, data)
            heights = env._grasp_state()['finger_tip_heights']
            self.assertTrue(np.all(np.asarray(heights) > 0.))
            data.qpos[20] += .5
            mujoco.mj_forward(model, data)
            np.testing.assert_allclose(env._grasp_state()['finger_tip_heights'], heights, atol=1e-8)
            data.qpos[21+np.array([0, 2, 3])] = [-.25, -.4, -.3]
            mujoco.mj_forward(model, data)
            self.assertLess(env._grasp_state()['finger_tip_heights'][0], -.03)
        finally:
            env.close()

    def test_prepare_cost_relaxes_after_contact_and_new_terms_are_opt_in(self):
        env = self.make()
        try:
            obs, info = env.reset(seed=0)
            env.Dcmm.data.qpos[21+MOVABLE] = np.array([-.25, -.4, -.3]+[.3, .2, .2]*3)
            env._last_grasp_state.update(distance=.05, finger_tip_heights=[-.05, .02, .02])
            env._sample_count = 20
            env._contact_ticks = 0
            ctrl = dict(base=np.zeros(2), arm=np.zeros(6), hand=np.zeros(12))
            info.update(bounce_grasp_terminal=False, success=False)
            env.compute_reward(obs, info, ctrl)
            self.assertLess(info['bounce_grasp_reward_terms']['finger_backfold'], 0.)
            self.assertLess(info['bounce_grasp_reward_terms']['finger_prepare_sync'], 0.)
            env._contact_ticks = 20
            env._finger_coverage_sum = 20.
            env._thumb_opposition_sum = 20.
            env.compute_reward(obs, info, ctrl)
            self.assertEqual(info['bounce_grasp_reward_terms']['finger_prepare_sync'], 0.)
            self.assertEqual(info['bounce_grasp_potential_terms']['finger_coverage'], 1.)
            self.assertEqual(info['bounce_grasp_potential_terms']['thumb_opposition'], .5)
        finally:
            env.close()
        env = self.make(options={})
        try:
            obs, info = env.reset(seed=0)
            info.update(bounce_grasp_terminal=False, success=False)
            env.compute_reward(obs, info, dict(base=np.zeros(2), arm=np.zeros(6), hand=np.zeros(12)))
            self.assertNotIn('finger_backfold', info['bounce_grasp_reward_terms'])
            self.assertNotIn('finger_coverage', info['bounce_grasp_potential_terms'])
            self.assertEqual(info['bounce_grasp_reward_profile'], 'base_v1')
        finally:
            env.close()

    def test_all_three_balls_run_with_limits_and_finite_rewards(self):
        for ball in ('default', 'ycb:tennis_ball', 'ycb:baseball'):
            env = self.make(ball)
            try:
                obs, info = env.reset(seed=0)
                self.assertEqual(info['bounce_grasp_reward_profile'], 'finger_coordination_v2')
                for _ in range(12):
                    obs, reward, terminated, truncated, info = env.step(dict(
                        base=np.zeros(2), arm=np.zeros(6), hand=-.08*np.ones(12)))
                    self.assertTrue(np.isfinite(reward))
                    self.assertTrue(np.all(np.isfinite(env.Dcmm.data.qpos)))
                    self.assertEqual(info['bounce_grasp_target_violation'], 0.)
                    self.assertLess(info['bounce_grasp_joint_violation'], .03)
                    if terminated or truncated:
                        break
            finally:
                env.close()


if __name__ == '__main__':
    unittest.main()
