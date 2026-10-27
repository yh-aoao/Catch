"""Synthetic fixtures verify loading/fitting, never constitute real calibration."""
import json
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from ycb_objects.materials import load, fit_drop, SCHEMA
from ycb_objects.calibration import template


def prior(ratio, friction):
    return dict(bounce_status='simulation_prior', friction_status='simulation_prior',
                damping_ratio=ratio, time_constant=.008, sliding_friction=friction,
                torsional_friction=.005)


class MaterialTests(unittest.TestCase):
    def test_distinct_parameters_follow_object_on_every_reset(self):
        from gym_dcmm.envs.bounce_physical_v2 import make_bounce_physical_v2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'synthetic.json'
            data = dict(schema=SCHEMA, surface_id='synthetic_test_only',
                        profiles=dict(tennis_ball=prior(.1,.2), baseball=prior(.2,.6)))
            data['profiles']['tennis_ball']['inertia_factor'] = .55
            path.write_text(json.dumps(data))
            loaded = load(path)
            self.assertEqual(len(loaded['sha256']), 64)
            for task in ('Tracking', 'Catching'):
                env = make_bounce_physical_v2(task=task, object_motion='bounce',
                    bounce_ball_mode='mixed', bounce_material_file=path,
                    viewer=False, render_mode=None, object_eval=False)
                try:
                    seen = set()
                    for seed in range(12):
                        env.reset(seed=seed)
                        profile = env.bounce_v2_episode['profile']
                        seen.add(profile)
                        expected = data['profiles'][profile]
                        m = env.Dcmm.model
                        pair = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_PAIR,'bounce_v2_ground')
                        self.assertAlmostEqual(m.pair_solref[pair,1],expected['damping_ratio'])
                        self.assertAlmostEqual(m.pair_friction[pair,0],expected['sliding_friction'])
                        if profile == 'tennis_ball':
                            bid=m.body('object').id
                            radius=float(m.geom_size[env.object_id,0])
                            np.testing.assert_allclose(m.body_inertia[bid],.55*m.body_mass[bid]*radius**2)
                    self.assertEqual(seen, {'tennis_ball','baseball'})
                finally:
                    env.close()

    def test_empty_measurements_are_rejected(self):
        with self.assertRaises(ValueError):
            fit_drop(template('tennis_ball'))

    def test_recovers_synthetic_normal_bounce_parameter(self):
        from ycb_objects.bounce_stability import drop
        from gym_dcmm.envs.bounce_physical_v2 import DT, profiles
        mass, radius, _ = profiles()['tennis_ball']
        data = template('tennis_ball')
        data['specimen_id'] = 'SYNTHETIC_TEST_ONLY'
        data['surfaces']['floor'] = 'synthetic_test_only'
        data['bounce_trials'] = []
        for i,h in enumerate((.5,.8,1.)):
            result = drop(mass,radius,h+radius,DT,time_constant=.008,damping_ratio=.17,
                          joint_damping=0,armature=0,vertical_velocity=0,duration=2.)
            data['bounce_trials'].append(dict(trial_id=str(i),surface_id='floor',
                release_clearance_m=h,rebound_clearance_m=result['max_height_after_first_floor']-radius))
        fitted = fit_drop(data)
        self.assertLess(abs(fitted['damping_ratio']-.17),.01)
        self.assertTrue(fitted['validation']['passed'])
        self.assertEqual(fitted['friction_status'],'simulation_prior')
        self.assertFalse(fitted['validation']['held_out_validated'])

    def test_rejects_fake_calibration_and_invalid_contact_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bad.json'
            for p in (dict(prior(.1,.3),bounce_status='fitted_drop'),prior(-1,.3),prior(.1,-1)):
                path.write_text(json.dumps(dict(schema=SCHEMA,surface_id='test',profiles=dict(tennis_ball=p))))
                with self.assertRaises(ValueError):
                    load(path)


if __name__ == '__main__':
    unittest.main()
