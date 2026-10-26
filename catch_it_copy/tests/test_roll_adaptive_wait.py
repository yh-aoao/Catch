"""Prediction, reward-incentive and real-MuJoCo parity tests for adaptive waiting."""
from pathlib import Path
import random
import tempfile
import unittest

import mujoco
import numpy as np

from gym_dcmm.envs.roll_adaptive import AdaptiveRollReward, make_roll_adaptive
from gym_dcmm.envs.roll_compat import make_roll_645edc4
from gym_dcmm.utils.roll_adaptive_wait import AdaptiveWaitConfig, interception, terms


class AdaptiveRewardTests(unittest.TestCase):
    def setUp(self):
        self.cfg = AdaptiveWaitConfig()

    def predict(self, velocity):
        return interception([0., 2.4, .74], velocity, .04, .9, .7, -9.81, self.cfg)

    def test_goal_follows_direction_speed_and_ballistic_height(self):
        slow, a = self.predict([0., -.5, 0.])
        fast, b = self.predict([0., -1.2, 0.])
        self.assertGreater(fast[2], slow[2])
        self.assertGreaterEqual(slow[2], self.cfg.min_height-1e-12)
        self.assertLessEqual(fast[2], self.cfg.max_height+1e-12)
        for goal, prediction in [(slow, a), (fast, b)]:
            t = prediction['fall_time']
            self.assertAlmostEqual(goal[2], .74-.5*9.81*t*t)
        left, _ = self.predict([-.3, -.8, 0.])
        right, _ = self.predict([.3, -.8, 0.])
        self.assertAlmostEqual(left[0], -right[0])
        self.assertLess(left[0], 0.)
        stationary, state = self.predict([0., 0., 0.])
        self.assertTrue(np.isfinite(stationary).all())
        self.assertLess(stationary[1], .9)
        self.assertEqual(state['exit_time'], 0.)

    def reward(self, palm=(0., .65, .5), previous=None, velocity=(0., 0., 0.),
               goal=(0., .65, .5), waiting=True, front=.80, hand=False, table=True):
        return terms(palm, previous, np.asarray(velocity), goal, waiting, front,
                     .9, hand, table, self.cfg)

    def test_waiting_prefers_aligned_settled_clear_hand(self):
        good = self.reward()
        self.assertGreater(good['ready'], self.reward(velocity=(.5, 0., 0.))['ready'])
        self.assertGreater(good['ready'], self.reward(palm=(.4, .65, .5))['ready'])
        intrusion = self.reward(front=1.0)
        self.assertEqual(intrusion['ready'], 0.)
        self.assertEqual(intrusion['xy'], 0.)
        self.assertLess(intrusion['hand_clearance'], good['hand_clearance'])
        self.assertLess(sum(self.reward(hand=True).values()), sum(good.values()))
        self.assertGreater(sum(good.values()), sum(self.reward(palm=(0., .65, .2)).values()))

    def test_goal_drift_does_not_pay_progress_and_release_removes_wait_penalties(self):
        palm = [0., .65, .5]
        for goal in ([.3, .7, .4], [-.3, .8, .6]):
            self.assertEqual(self.reward(palm=palm, previous=palm, goal=goal)['progress'], 0.)
        released = self.reward(waiting=False, front=1., hand=True, table=True)
        self.assertEqual(released['hand_clearance'], 0.)
        self.assertEqual(released['ready'], 0.)
        self.assertEqual(released['early_contact'], 0.)

    def test_parameter_validation(self):
        for options in [dict(min_height=.7, max_height=.6), dict(xy_scale=0),
                        dict(hand_weight=-1), dict(max_height=float('nan'))]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                AdaptiveWaitConfig(**options)


class AdaptiveRuntimeTests(unittest.TestCase):
    def parameters(self, task):
        return dict(task=task, object_motion='roll', viewer=False, object_eval=False,
                    render_mode=None, steps_per_policy=20, env_time=.12)

    def test_same_actions_produce_identical_physics_termination_and_observation(self):
        action = dict(base=np.array([.1, .2]), arm=np.zeros(6), hand=np.zeros(12))
        for task in ('Tracking', 'Catching'):
            parameters = self.parameters(task)
            np.random.seed(123)
            random.seed(123)
            baseline = make_roll_645edc4(parameters)
            try:
                np.random.seed(456)
                random.seed(456)
                baseline.reset()
                original = []
                for _ in range(5):
                    obs, reward, term, trunc, info = baseline.step(action)
                    original.append((baseline.Dcmm.data.qpos.copy(), baseline.Dcmm.data.qvel.copy(),
                                     obs['object']['pos3d'].copy(), reward, term, trunc))
                    if term or trunc:
                        break
            finally:
                baseline.close()
            np.random.seed(123)
            random.seed(123)
            revised = make_roll_adaptive(**parameters)
            try:
                with tempfile.TemporaryDirectory() as directory:
                    revised.set_roll_reward_log_dir(directory)
                    np.random.seed(456)
                    random.seed(456)
                    revised.reset()
                    for qpos, qvel, object_obs, old_reward, term, trunc in original:
                        obs, reward, new_term, new_trunc, info = revised.step(action)
                        np.testing.assert_array_equal(revised.unwrapped.Dcmm.data.qpos, qpos)
                        np.testing.assert_array_equal(revised.unwrapped.Dcmm.data.qvel, qvel)
                        np.testing.assert_array_equal(obs['object']['pos3d'], object_obs)
                        self.assertEqual((term, trunc), (new_term, new_trunc))
                        state = info['roll_adaptive_wait']
                        self.assertAlmostEqual(reward, old_reward-state['removed']+state['added'])
                        self.assertEqual(info['roll_reward_version'], 'adaptive_wait_v1')
                    self.assertTrue(list(Path(directory).glob('*.jsonl')))
            finally:
                revised.close()

    def test_ycb_release_reset_and_reject_hard_guard(self):
        env = make_roll_adaptive(test_object='ycb:baseball', **self.parameters('Catching'))
        try:
            env.reset()
            raw = env.unwrapped
            model, data = raw.Dcmm.model, raw.Dcmm.data
            bid = model.body('object').id
            q = model.jnt_qposadr[model.body_jntadr[bid]]
            raw.object_throw = True
            data.qpos[q:q+3] = [0., .7, .7]
            mujoco.mj_forward(model, data)
            _, _, _, _, info = env.step(dict(base=np.zeros(2), arm=np.zeros(6), hand=np.zeros(12)))
            self.assertFalse(info['roll_waiting'])
            self.assertEqual(info['roll_wait_terms']['ready'], 0.)
            env.reset()
            self.assertFalse(env.left_table)
            self.assertIsNone(env.previous_palm)
            self.assertEqual(env.sums, {})
        finally:
            env.close()
        with self.assertRaises(ValueError):
            make_roll_adaptive(object_motion='roll', roll_wait_guard=True)

    def test_registered_vector_factory_can_assign_spec_and_forward_attributes(self):
        import gymnasium as gym
        import gym_dcmm
        env = gym.make_vec('gym_dcmm/RollAdaptiveWorld-v0', num_envs=1,
                           vectorization_mode='sync', **self.parameters('Catching'))
        try:
            env.reset()
            self.assertEqual(env.call('object_motion')[0], 'roll')
            _, reward, _, _, info = env.step(dict(base=np.zeros((1, 2)),
                                                  arm=np.zeros((1, 6)), hand=np.zeros((1, 12))))
            self.assertTrue(np.isfinite(reward).all())
            self.assertEqual(info['roll_reward_version'][0], 'adaptive_wait_v1')
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
