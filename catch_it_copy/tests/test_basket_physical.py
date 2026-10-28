import unittest
import numpy as np
from gym_dcmm.utils.basket_physical import release_quality


class PhysicalTests(unittest.TestCase):
    def test_release_requires_forward_transport_and_proximal_speed(self):
        self.assertEqual(release_quality({}), 0.)
        self.assertEqual(release_quality(dict(arm_drive_forward=.06, arm_drive_speed=0.)), 0.)
        self.assertEqual(release_quality(dict(arm_drive_forward=0., arm_drive_speed=.3)), 0.)
        self.assertEqual(release_quality(dict(arm_drive_forward=-.1, arm_drive_speed=.3)), 0.)
        self.assertEqual(release_quality(dict(arm_drive_forward=.06, arm_drive_speed=.3)), 1.)
        self.assertEqual(release_quality(dict(arm_drive_forward=.03, arm_drive_speed=.6)), .5)

    def test_last_contact_controls_quality_not_best_previous_swing(self):
        state = dict(arm_drive_forward=.001, arm_drive_speed=.01, best_arm_drive=1.)
        self.assertLess(release_quality(state), .02)

    def test_actual_flight_records_crossing_but_requires_qualified_success(self):
        from tests.test_basket_catching import fixture, scope
        from tests.test_basket_mobile import config
        from gym_dcmm.utils.basket_simple import update_launch
        scope['update_simple_launch'] = update_launch
        for quality in (0., .5, 1.):
            env, _, _ = fixture()
            env.mobile_basket = False
            env.basket_fixed_physical = True
            env.basket_fixed_arm_drive_bonus = 0.
            env.basket_reward_version = 'basket_aim_v1'
            env.basket_cfg = config()
            env.basket_cfg.basket_floor_z = 0.
            env.basket_cfg.basket_radius = .2
            env.basket_cfg.basket_ball_radius = .04
            env.basket_center = np.array([0., 1., .55])
            env.basket_phase = 'flight'
            env.basket_had_hand_contact = True
            env.start_time = 0.
            env.basket_aim_hit_time = None
            env.basket_simple_state = dict(confirmed=True, physical_release_quality=quality,
                candidate=dict(time=.4, position=np.array([0., .4, .6]),
                               direction=np.array([0.,1.]), invalid=False))
            env.Dcmm.data.time = 1.
            env.Dcmm.data.qpos[37:40] = [0., 1., .54]
            scope['_basket_check_flight'](env, np.array([0., 1., .56]))
            self.assertEqual(env.basket_aim_hit_time, 1.)
            self.assertFalse(env.terminated)
            env.Dcmm.data.time = 1.31
            scope['_basket_check_flight'](env, env.Dcmm.data.qpos[37:40].copy())
            self.assertEqual(env.terminated_reason, 'basket_score' if quality == 1. else 'basket_unqualified')


if __name__ == '__main__':
    unittest.main()
