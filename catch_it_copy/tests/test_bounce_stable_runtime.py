"""Verify opt-in physics routing, reset persistence, and unchanged timing/reward."""
import random
import unittest
from unittest.mock import patch

import numpy as np
import mujoco

from gym_dcmm.envs.bounce_stable import make_bounce_stable, TIMESTEP, SOLREF


class StableRuntimeTests(unittest.TestCase):
    def test_routes_reset_parameters_timing_and_reward(self):
        for task in ('Tracking', 'Catching'):
            for object_name in ('default', 'ycb:tennis_ball', 'ycb:baseball'):
                with self.subTest(task=task, object=object_name):
                    np.random.seed(123)
                    random.seed(123)
                    env = make_bounce_stable(task=task, object_motion='bounce',
                        test_object=object_name, viewer=False, render_mode=None,
                        object_eval=False, steps_per_policy=20, env_time=.2)
                    try:
                        for _ in range(2):
                            obs, info = env.reset(seed=123)
                            base = env.unwrapped
                            model, data = base.Dcmm.model, base.Dcmm.data
                            self.assertAlmostEqual(model.opt.timestep, TIMESTEP)
                            self.assertEqual(model.opt.integrator, mujoco.mjtIntegrator.mjINT_RK4)
                            for gid in (base.object_id, base.floor_id):
                                np.testing.assert_allclose(model.geom_solref[gid], SOLREF)
                            self.assertEqual(base.fps, 25.)
                            before = data.time
                            action = dict(base=np.zeros(2), arm=np.zeros(6), hand=np.zeros(12))
                            result = env.step(action)
                            self.assertAlmostEqual(data.time-before, .04, places=10)
                            self.assertTrue(np.isfinite(data.qpos).all())
                            self.assertEqual(result[-1]['bounce_physics_version'], 'stable_v1')
                    finally:
                        env.close()

    def test_wrapper_preserves_reward_and_flags(self):
        # The physical environment test above checks real transitions; here
        # explicitly verify the wrapper never reshapes the underlying reward.
        env = make_bounce_stable(task='Catching', object_motion='bounce',
            viewer=False, render_mode=None, object_eval=False)
        try:
            with patch.object(env.env, 'step', return_value=({}, 12.345, True, False, {})):
                self.assertEqual(env.step({})[1:4], (12.345, True, False))
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
