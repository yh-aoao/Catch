import unittest
from gym_dcmm.utils.basket_launch import direct_launch_progress, retention_credit


class DirectLaunchTests(unittest.TestCase):
    def progress(self, state, velocity):
        return direct_launch_progress(state, velocity, [0, 2, 2], [0, 0, 0], [0, 2, 1])

    def test_partial_directions_and_budget(self):
        state = {}
        self.assertAlmostEqual(self.progress(state, [0, 2, 0]), .25)
        self.assertAlmostEqual(self.progress(state, [0, 0, 2]), .25)
        self.assertAlmostEqual(self.progress(state, [0, 2, 2]), .5)
        self.assertEqual(self.progress(state, [0, 2, 2]), 0.)
        self.assertEqual(self.progress({}, [0, -2, -2]), 0.)

    def test_support_budget_and_recontact(self):
        state = {}
        total = sum(retention_credit(state, [0, 0, 0], .01)[0] for _ in range(200))
        self.assertAlmostEqual(total, 1.)
        state.pop('v5_local_position')
        self.assertEqual(retention_credit(state, [0, 0, 0], .01)[0], 0.)
        self.assertEqual(retention_credit(state, [0, 0, 0], .01)[0], 0.)

    def test_slipping_gets_less_support(self):
        state = {}
        retention_credit(state, [0, 0, 0], .01)
        credit, speed = retention_credit(state, [0, .01, 0], .01)
        self.assertAlmostEqual(speed, 1.)
        self.assertLess(credit, .001)


if __name__ == '__main__':
    unittest.main()
