"""Synthetic data validates analysis math; it is not real-ball calibration."""
import unittest
import numpy as np

from ycb_objects.calibration import analyze, template, fit_rolling


class CalibrationTests(unittest.TestCase):
    def test_empty_template_has_no_invented_physics(self):
        result = analyze(template('baseball'))
        self.assertEqual(result['status'], 'awaiting_measurements')
        self.assertIsNone(result['mass_kg'])
        self.assertIsNone(result['sphere_radius_m'])
        self.assertFalse(result['mujoco_parameters_identified'])
        self.assertFalse(result['inertia_identified'])

    def test_units_and_height_restitution(self):
        data = template('baseball')
        data.update(mass_g=140., diameters_mm=[72., 73., 74.])
        data['bounce_trials'][0].update(release_clearance_m=.5, rebound_clearance_m=.32)
        result = analyze(data)
        self.assertAlmostEqual(result['mass_kg'], .14)
        self.assertAlmostEqual(result['sphere_radius_m'], .0365)
        self.assertAlmostEqual(result['bounce'][0]['effective_restitution_height_ratio'], .8)
        self.assertFalse(result['mujoco_parameters_identified'])

    def test_recovers_known_rolling_trajectory(self):
        trial = template('tennis_ball')['rolling_trials'][0]
        trial['samples'] = [dict(t_s=float(t+10), distance_m=float(.2+.7*t-.05*t*t))
                            for t in np.linspace(0., 2., 9)]
        result = fit_rolling(trial)
        self.assertAlmostEqual(result['initial_speed_m_s'], .7)
        self.assertAlmostEqual(result['acceleration_m_s2'], -.1)
        self.assertAlmostEqual(result['stopping_distance_extrapolation_m'], 2.45)
        self.assertLess(result['position_rmse_m'], 1e-10)

    def test_rejects_invalid_or_stopped_records(self):
        for mass in [-1, float('nan'), True]:
            data = template('baseball')
            data['mass_g'] = mass
            with self.subTest(mass=mass), self.assertRaises(ValueError):
                analyze(data)
        trial = template('baseball')['rolling_trials'][0]
        trial['samples'] = [dict(t_s=t, distance_m=x) for t, x in [(0, 0), (1, .3), (2, .4), (3, .4)]]
        with self.assertRaises(ValueError):
            fit_rolling(trial)
        data = template('baseball')
        data['bounce_trials'][0].update(release_clearance_m=.3, rebound_clearance_m=.5)
        with self.assertRaises(ValueError):
            analyze(data)


if __name__ == '__main__':
    unittest.main()
