"""Basket YCB dimensions and contact identity must exist before preparation."""
import json
import unittest

import mujoco
import numpy as np

from configs.env import DcmmCfg
from gym_dcmm.envs.basket_object_test import PRIORS, make_basket_object_test


class BasketObjectTests(unittest.TestCase):
    def make(self, name):
        return make_basket_object_test('ycb:' + name, task='Catching',
            object_motion='throw_basket', render_mode=None, viewer=False,
            imshow_cam=False, object_eval=False,
            basket_fixed_base_training=True, basket_reward_version='basket_aim_v1',
            basket_arm_control='joint_velocity', basket_fixed_physical=True,
            basket_carry_reward=True, basket_fixed_hand_flex=[.8, .7, .7])

    def test_real_collision_sphere_and_instance_local_parameters(self):
        original = (DcmmCfg.basket_ball_radius, DcmmCfg.basket_ball_mass)
        profiles = json.loads(PRIORS.read_text())['profiles']
        for name in ('tennis_ball', 'baseball'):
            env = self.make(name)
            try:
                model, profile = env.Dcmm.model, profiles[name]
                sphere = env.object_id
                body = int(model.geom_bodyid[sphere])
                self.assertEqual(model.geom_type[sphere], mujoco.mjtGeom.mjGEOM_SPHERE)
                self.assertTrue(model.geom_contype[sphere] or model.geom_conaffinity[sphere])
                collision = [i for i in range(model.ngeom) if model.geom_bodyid[i] == body
                             and (model.geom_contype[i] or model.geom_conaffinity[i])]
                self.assertEqual(collision, [sphere])
                self.assertAlmostEqual(model.geom_size[sphere, 0], profile['radius_m'])
                self.assertAlmostEqual(model.body_mass[body], profile['mass_kg'])
                inertia = profile['inertia_factor'] * profile['mass_kg'] * profile['radius_m']**2
                np.testing.assert_allclose(model.body_inertia[body], inertia)
                np.testing.assert_array_equal(model.body_ipos[body], 0)
                joint = model.body_jntadr[body]
                dof = model.jnt_dofadr[joint]
                np.testing.assert_array_equal(model.dof_armature[dof:dof+6], 0)
                np.testing.assert_array_equal(model.dof_damping[dof:dof+6], 0)
                self.assertEqual(env.basket_cfg.basket_ball_radius, profile['radius_m'])
                self.assertEqual(env.basket_cfg.basket_ball_mass, profile['mass_kg'])
            finally:
                env.close()
        self.assertEqual((DcmmCfg.basket_ball_radius, DcmmCfg.basket_ball_mass), original)

    def test_reject_incompatible_task_and_collision_override(self):
        with self.assertRaises(ValueError):
            make_basket_object_test('ycb:tennis_ball', object_motion='throw_basket', task='Tracking')
        with self.assertRaises(ValueError):
            make_basket_object_test('ycb:baseball', object_motion='throw_basket', task='Catching',
                basket_fixed_base_training=True, basket_fixed_physical=True,
                basket_reward_version='basket_aim_v1', ycb_collision='mesh')


if __name__ == '__main__':
    unittest.main()
