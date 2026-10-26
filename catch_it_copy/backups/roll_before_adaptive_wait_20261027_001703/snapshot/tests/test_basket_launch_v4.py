import unittest
from gym_dcmm.utils.basket_launch import coupled_readiness, confirm_throw


class LaunchV4Tests(unittest.TestCase):
    def test_direction(self):
        args = ([0, 2, 2], [0, 0, 0], [0, 2, 1])
        self.assertEqual(coupled_readiness([0, -2, -2], *args), 0)
        self.assertEqual(coupled_readiness([0, 0, 2], *args), 0)
        self.assertAlmostEqual(coupled_readiness([0, 2, 2], *args), 1)

    def test_flight(self):
        c = dict(time=0., position=[0, 0, 0], direction=[0, 1],
                 forward=1., upward=1., eligible=True)
        self.assertFalse(confirm_throw(c, [0, .2, 0], .05, False)[0])
        self.assertFalse(confirm_throw(c, [0, .01, 0], .2, False)[0])
        self.assertTrue(confirm_throw(c, [0, .2, 0], .2, False)[0])
        self.assertFalse(confirm_throw(c, [0, .2, 0], .2, True)[0])
        self.assertFalse(confirm_throw(c, [0, .3, 0], .3, False)[0])

    def test_drop(self):
        c = dict(time=0., position=[0, 0, 0], direction=[0, 1],
                 forward=0., upward=-1., eligible=False)
        self.assertFalse(confirm_throw(c, [0, .2, 0], .3, False)[0])
