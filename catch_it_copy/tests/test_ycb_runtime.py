"""MuJoCo integration checks for the opt-in YCB evaluation controls."""
import unittest

import numpy as np

from gym_dcmm.envs.object_test import make_object_test
from ycb_objects.roll_probe import model_for, run


class YCBRuntimeTests(unittest.TestCase):
    def test_roll_speed_is_applied_on_each_reset_with_matching_spin(self):
        for name in ('tennis_ball', 'baseball'):
            for task in ('Tracking', 'Catching'):
                for speed in (.5, [.3, .6]):
                    env = make_object_test('ycb:' + name, task=task, object_motion='roll',
                                           object_eval=False, viewer=False, render_mode=None,
                                           roll_speed=speed)
                    try:
                        for _ in range(3):
                            _, info = env.reset()
                            v = info['roll_initial_velocity']
                            measured = np.linalg.norm(v[:2])
                            if isinstance(speed, float):
                                self.assertAlmostEqual(measured, speed)
                            else:
                                self.assertGreaterEqual(measured, speed[0])
                                self.assertLessEqual(measured, speed[1])
                            radius = env.Dcmm.model.geom_size[env.object_id, 0]
                            np.testing.assert_allclose(v[3:5], [-v[1]/radius, v[0]/radius])
                            self.assertEqual(v[2], 0.)
                            self.assertEqual(v[5], 0.)
                            self.assertLess(v[1], 0.)
                            self.assertGreaterEqual(env.object_pos3d[1], 2.1)
                            self.assertLessEqual(env.object_pos3d[1], 2.6)
                        env.step(dict(base=np.zeros(2), arm=np.zeros(6), hand=np.zeros(12)))
                    finally:
                        env.close()

    def test_invalid_roll_speeds_rejected(self):
        from gym_dcmm.utils.roll_speed import speed_range
        for value in (-1, float('nan'), float('inf'), True, [.8, .3], [1], [0, 1, 2], '0.5'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                speed_range(value)
        self.assertEqual(speed_range(0), (0., 0.))
        self.assertIsNone(speed_range(None))
        with self.assertRaises(ValueError):
            make_object_test('ycb:baseball', object_motion='throw', roll_speed=.5)

    def test_roll_reset_preserves_inertia_and_requested_damping(self):
        constants = []
        for damping in (None, 0.):
            np.random.seed(123)
            env = make_object_test('ycb:tennis_ball', task='Catching', object_motion='roll',
                                   object_eval=False, viewer=False, render_mode=None,
                                   env_time=5., ycb_joint_damping=damping)
            try:
                for _ in range(2):
                    _, info = env.reset()
                    model = env.Dcmm.model
                    bid = int(model.geom_bodyid[env.object_id])
                    expected = .0001 if damping is None else damping
                    np.testing.assert_allclose(info['test_object_joint_damping'], expected)
                    self.assertEqual(info['test_object_env_time'], 5.)
                    self.assertAlmostEqual(model.body_mass[bid], .058)
                    self.assertEqual(model.geom_contype[env.object_id], 0)
                    ids = np.flatnonzero((model.geom_bodyid == bid) & (model.geom_contype != 0))
                    self.assertEqual(len(ids), 1)
                    for attribute in ('geom_friction', 'geom_solref', 'geom_solimp', 'geom_condim'):
                        np.testing.assert_allclose(getattr(model, attribute)[ids[0]],
                                                   getattr(model, attribute)[env.object_id])
                    constants.append(model.body_inertia[bid].copy())
                obs, _, _, _, _ = env.step(dict(base=np.zeros(2), arm=np.zeros(6), hand=np.zeros(12)))
                self.assertTrue(np.isfinite(obs['object']['pos3d']).all())
            finally:
                env.close()
        for inertia in constants[1:]:
            np.testing.assert_allclose(inertia, constants[0])

    def test_ball_defaults_use_sphere_and_match_original_inertia(self):
        import mujoco
        from ycb_objects.loader import sphere_radius, load
        for name in ('tennis_ball', 'baseball'):
            reference, _ = model_for(name, 'baseline')
            env = make_object_test('ycb:' + name, task='Catching', object_motion='roll',
                                   object_eval=False, viewer=False, render_mode=None)
            try:
                for _ in range(2):
                    _, info = env.reset()
                    model = env.Dcmm.model
                    bid = model.body('object').id
                    ids = np.flatnonzero((model.geom_bodyid == bid) & (model.geom_contype != 0))
                    self.assertEqual(info['test_object_collision'], 'sphere')
                    self.assertEqual(len(ids), 1)
                    self.assertEqual(model.geom_type[ids[0]], mujoco.mjtGeom.mjGEOM_SPHERE)
                    self.assertAlmostEqual(model.geom_size[ids[0], 0], sphere_radius(load(name)))
                    np.testing.assert_allclose(model.body_inertia[bid], reference.body_inertia[reference.body('object').id])
                    np.testing.assert_allclose(model.body_ipos[bid], reference.body_ipos[reference.body('object').id])
                    self.assertEqual(model.geom_contype[env.object_id], 0)
                    mesh_ids = np.flatnonzero((model.geom_bodyid == bid) &
                                             (model.geom_type == mujoco.mjtGeom.mjGEOM_MESH))
                    self.assertTrue(len(mesh_ids))
                    self.assertTrue(np.all(model.geom_contype[mesh_ids] == 0))
                env.step(dict(base=np.zeros(2), arm=np.zeros(6), hand=np.zeros(12)))
            finally:
                env.close()

    def test_invalid_damping_rejected(self):
        for name, value in [('ycb:tennis_ball', -1), ('ycb:tennis_ball', float('nan')),
                            ('ycb:tennis_ball', float('inf')), ('default', 0.)]:
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                make_object_test(name, ycb_joint_damping=value)

    def test_sphere_control_really_collides_and_preserves_mesh_inertia(self):
        mesh, radius = model_for('tennis_ball', 'baseline')
        sphere, _ = model_for('tennis_ball', 'sphere')
        np.testing.assert_allclose(mesh.body_inertia[mesh.body('object').id],
                                   sphere.body_inertia[sphere.body('object').id])
        np.testing.assert_allclose(mesh.body_ipos[mesh.body('object').id],
                                   sphere.body_ipos[sphere.body('object').id])
        rows, _ = run(sphere, radius, .4, .5)
        self.assertTrue(any(r['contacts'] > 0 for r in rows))
        self.assertTrue(all(r['z'] > .7 for r in rows))
        self.assertTrue(any('table_surface' in r['pairs'] for r in rows))


if __name__ == '__main__':
    unittest.main()
