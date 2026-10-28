import unittest
from types import SimpleNamespace as N

import numpy as np

from gym_dcmm.utils.basket_grip import cup_joints, arm_joints, validate_joints
from tests.test_basket_mobile import config


class GripTests(unittest.TestCase):
    def test_default_and_deeper_cup_preserve_spread_and_thumb(self):
        original = cup_joints()
        deeper = cup_joints(config(hold_hand_flex=[.8, .7, .7]))
        np.testing.assert_array_equal(original[[0, 2, 3]], [.6, .5, .5])
        for first in (0, 4, 8):
            np.testing.assert_array_equal(deeper[[first, first+2, first+3]], [.8, .7, .7])
        np.testing.assert_array_equal(deeper[[1, 5, 9, 12]], 0.)
        np.testing.assert_array_equal(deeper[13:], original[13:])

    def test_offsets_do_not_mutate_source_pose(self):
        base = np.zeros(6)
        pose = arm_joints(base, config(hold_arm_offset_deg=[0, 0, 0, 0, 20, 0]))
        self.assertAlmostEqual(pose[4], np.radians(20))
        np.testing.assert_array_equal(base, 0.)

    def test_invalid_pose_vectors_and_joint_limits_rejected(self):
        for options in ({'hold_hand_flex': [1, 2]}, {'hold_ball_local': [0, 0, np.nan]},
                        {'hold_arm_offset_deg': [0]*5}):
            with self.assertRaises(ValueError):
                config(**options)
        model = N(jnt_qposadr=np.array([0]), jnt_limited=np.array([True]),
                  jnt_range=np.array([[0., 1.]]), joint=lambda _: N(name='finger'))
        validate_joints(model, 0, [.8])
        with self.assertRaises(ValueError):
            validate_joints(model, 0, [1.1])


if __name__ == '__main__':
    unittest.main()
