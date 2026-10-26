"""Check the optional limiter against real robot/table geometry and reset state."""
import unittest

import mujoco
import numpy as np

from gym_dcmm.envs.object_test import make_object_test


class RollWaitGuardTests(unittest.TestCase):
    def make_env(self, task='Catching', enabled=True):
        np.random.seed(0)
        return make_object_test('ycb:baseball', task=task, object_motion='roll',
                                object_eval=False, viewer=False, render_mode=None,
                                steps_per_policy=20, env_time=5., roll_speed=.5,
                                roll_wait_guard=enabled)

    def test_filters_early_dive_and_keeps_hand_outside_table(self):
        env = self.make_env()
        try:
            env.reset()
            risky = dict(base=np.array([0., 1.5]), arm=np.array([0., .025, -.025, .02, .02, .02]),
                         hand=np.full(12, .15))
            snapshots = {key: value.copy() for key, value in risky.items()}
            filtered = env._roll_wait_guard.filter(risky)
            np.testing.assert_array_equal(filtered['arm'][3:], 0.)
            np.testing.assert_array_equal(filtered['hand'], 0.)
            for key in risky:
                np.testing.assert_array_equal(risky[key], snapshots[key])
            for _ in range(25):
                _, _, term, trunc, info = env.step(risky)
                guard = info['roll_wait_guard']
                self.assertTrue(guard['active'])
                self.assertTrue(guard['intervention'])
                self.assertLess(guard['hand_front_y'], guard['edge_y'])
                self.assertFalse(term or trunc)
        finally:
            env.close()

    def test_release_requires_whole_ball_clear_and_resets_next_episode(self):
        env = self.make_env()
        try:
            env.reset()
            guard = env._roll_wait_guard
            model, data = env.Dcmm.model, env.Dcmm.data
            bid = model.body('object').id
            q = model.jnt_qposadr[model.body_jntadr[bid]]
            radius = model.geom_size[env.object_id, 0]
            action = dict(base=np.zeros(2), arm=np.ones(6)*.01, hand=np.ones(12)*.01)
            # Center is outside, but the rear of the ball has not cleared.
            data.qpos[q:q+3] = [0., .9 - radius*.5, .9]
            mujoco.mj_forward(model, data)
            guard.filter(action)
            self.assertTrue(guard.diagnostics['active'])
            data.qpos[q:q+3] = [0., .9 - radius - .02, .9]
            mujoco.mj_forward(model, data)
            self.assertIs(guard.filter(action), action)
            self.assertFalse(guard.diagnostics['active'])
            data.qpos[q+1] = 2.
            mujoco.mj_forward(model, data)
            self.assertIs(guard.filter(action), action)
            env.reset()
            env._roll_wait_guard.filter(action)
            self.assertTrue(env._roll_wait_guard.diagnostics['active'])
        finally:
            env.close()

    def test_track_and_disabled_interfaces(self):
        for task, enabled in [('Tracking', True), ('Catching', False)]:
            env = self.make_env(task=task, enabled=enabled)
            try:
                env.reset()
                _, _, _, _, info = env.step(dict(base=np.zeros(2), arm=np.zeros(6), hand=np.zeros(12)))
                self.assertEqual('roll_wait_guard' in info, enabled)
            finally:
                env.close()
        with self.assertRaises(ValueError):
            make_object_test('ycb:baseball', object_motion='throw', roll_wait_guard=True)


if __name__ == '__main__':
    unittest.main()
