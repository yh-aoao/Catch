import unittest
from types import SimpleNamespace
from gym_dcmm.utils.basket_simple import update_launch, reward


CFG = SimpleNamespace(basket_simple_free_seconds=.05, basket_simple_forward_speed=.3,
                      basket_simple_target_distance=.5, basket_simple_launch_bonus=5.,
                      basket_simple_progress_bonus=10., basket_simple_failure_cost=5.)


class SimpleLaunchTests(unittest.TestCase):
    def step(self, state, time, x=0., speed=1., touching=False, collision=False):
        update_launch(state, [x, 0, 1], [speed, 0, 0], [2, 0, 1],
                      time, touching, collision, CFG)

    def test_confirmation_and_capped_total(self):
        s = {}
        self.step(s, 0, touching=True)
        self.step(s, .04, .04)
        self.assertEqual(reward(s, False, CFG)[0], 0.)
        self.step(s, .06, .1)
        self.assertAlmostEqual(reward(s, False, CFG)[0], 7.)
        self.step(s, .1, .6)
        self.assertAlmostEqual(reward(s, False, CFG)[0], 8.)
        self.step(s, .2, 1.)
        self.assertEqual(reward(s, True, CFG)[0], 0.)

    def test_drop_backward_and_no_prior_contact(self):
        for speed in (0., -.5, .29):
            s = {}
            self.step(s, 0, speed=speed, touching=True)
            self.step(s, .1, .1)
            self.assertEqual(reward(s, True, CFG)[0], -5.)
            self.assertEqual(reward(s, True, CFG)[0], 0.)
        s = {}
        self.step(s, .1, .5)
        self.assertEqual(reward(s, True, CFG)[0], -5.)

    def test_collision_before_confirmation(self):
        s = {}
        self.step(s, 0, touching=True)
        self.step(s, .02, .02, collision=True)
        self.step(s, .1, .1)
        self.assertEqual(reward(s, True, CFG)[0], -5.)

    def test_recontact_or_collision_stops_credit(self):
        for contact in ('touching', 'collision'):
            s = {}
            self.step(s, 0, touching=True)
            self.step(s, .06, .1)
            self.assertEqual(reward(s, False, CFG)[0], 7.)
            self.step(s, .1, .2, **{contact: True})
            self.step(s, .2, .6)
            self.assertEqual(reward(s, True, CFG)[0], 0.)

    def test_retreat_does_not_pay_twice(self):
        s = {}
        self.step(s, 0, touching=True)
        self.step(s, .06, .2)
        self.assertEqual(reward(s, False, CFG)[0], 9.)
        self.step(s, .1, .1)
        self.step(s, .2, .2)
        self.assertEqual(reward(s, False, CFG)[0], 0.)


if __name__ == '__main__':
    unittest.main()
