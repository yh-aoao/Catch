"""Convergence check for an isolated passive sphere; not measured material data."""
import unittest
from ycb_objects.bounce_stability import drop


class BounceStabilityTests(unittest.TestCase):
    def test_stable_v1_convergence_and_passive_energy(self):
        from gym_dcmm.envs.bounce_stable import TIMESTEP, SOLREF
        for mass, radius in ((.05, .04), (.058, .0332758), (.138, .0361762)):
            for height in (.898, .9, .902):
                candidate = drop(mass, radius, height, TIMESTEP,
                                 time_constant=SOLREF[0], damping_ratio=SOLREF[1])
                reference = drop(mass, radius, height, TIMESTEP/2,
                                 time_constant=SOLREF[0], damping_ratio=SOLREF[1])
                self.assertLess(abs(candidate['max_height_after_first_floor']-
                                    reference['max_height_after_first_floor']), .015)
                self.assertFalse(candidate['exceeds_energy_height_bound'])
                self.assertLess(candidate['max_penetration'], .005)

    def test_refining_timestep_reduces_impact_phase_variation(self):
        values = []
        for dt in (.002, .00025):
            peaks = [drop(.058, .033275785333333335, h, dt)['max_height_after_first_floor']
                     for h in (.898, .9, .902)]
            values.append(max(peaks)-min(peaks))
        self.assertLess(values[1], values[0])

    def test_refined_drop_does_not_gain_height_above_initial_energy(self):
        result = drop(.138, .036176233833333335, .9, .00025)
        self.assertTrue(result['hit_floor'])
        self.assertFalse(result['exceeds_energy_height_bound'])
        with self.assertRaises(ValueError):
            drop(.1, .04, .9, 0.)


if __name__ == '__main__':
    unittest.main()
