"""Task invariants: release eligibility, bounded progress and physical interfaces."""
import unittest
from types import SimpleNamespace as N

import numpy as np

from gym_dcmm.utils import basket_mobile as mobile
from gym_dcmm.utils.basket_tracking import parking_state


def config(**options):
    return mobile.make_config(N(basket_fail_dist=2.5, basket_simple_launch_bonus=5.,
        basket_simple_failure_cost=5., basket_simple_smooth_weight=.01,
        basket_simple_free_seconds=.05, basket_simple_forward_speed=.3,
        basket_simple_target_distance=.5, basket_tilt_deg=0.), options)


def parking(distance=0., settled=True, speed=0.):
    return dict(distance=distance, speed=speed, settled=settled,
                velocity=np.array([0., speed]), desired_velocity=np.zeros(2))


class MobileTests(unittest.TestCase):
    def setUp(self):
        self.cfg = config()
        self.state, self.launch = mobile.new_state(.5), {}

    def update(self, now, touching=True, settled=True, speed=1., distance=0.):
        mobile.update_physics(self.state, self.launch, parking(distance, settled),
            [0., now-1., 1.], [0., speed, 0.], [0., 1., 0.], now, .02,
            touching, False, self.cfg)

    def unlock(self):
        for now in np.arange(.30, .501, .02):
            self.update(float(now))
        self.assertTrue(self.state['unlocked'])

    def test_configuration_isolated_and_validated(self):
        base = N(basket_fail_dist=2.5, basket_center=np.array([0., 2.2, .9]))
        cfg = mobile.make_config(base)
        np.testing.assert_array_equal(base.basket_center, [0., 2.2, .9])
        self.assertEqual(cfg.basket_center[2], .55)
        for options in ({'max_speed': 0}, {'max_time': float('nan')},
                        {'target_x_range': [1., -1.]}, {'target_y_range': [0., 1.]},
                        {'typo': 1}):
            with self.assertRaises(ValueError):
                mobile.make_config(base, options)

    def test_seeded_target_ranges_and_parking_tracks_both_axes(self):
        a, b = np.random.default_rng(123), np.random.default_rng(123)
        target = mobile.sample_target(self.cfg, a)
        np.testing.assert_array_equal(target, mobile.sample_target(self.cfg, b))
        self.assertTrue(-.4 <= target[0] <= .4 and 1.35 <= target[1] <= 1.85)
        self.cfg.basket_base_front_dist = .934
        p = parking_state(np.zeros(3), np.zeros(2), 0., target, self.cfg)
        np.testing.assert_allclose(p['target'][:2], target[:2]-[0., .934])
        p2 = parking_state(np.array([.2, .2, 0.]), np.zeros(2), 0., target, self.cfg)
        np.testing.assert_array_equal(p['target'], p2['target'])

    def test_settling_requires_continuous_contact_and_stationary_position(self):
        for now in np.arange(.30, .44, .02):
            self.update(float(now))
        self.update(.44, settled=False)
        self.assertEqual(self.state['settle_seconds'], 0.)
        self.assertFalse(self.state['unlocked'])
        self.unlock()
        self.assertEqual(self.state['ready_pending'], 3.)

    def test_initial_ball_settling_does_not_count_as_a_release(self):
        self.update(.02)
        self.update(.10, touching=False)
        self.assertIsNone(self.state['release_failure'])
        self.assertFalse(self.launch.get('confirmed', False))
        self.update(.31, touching=False)
        self.assertEqual(self.state['release_failure'], 'premature_release')

    def test_preparking_release_and_slow_drop_cannot_earn_throw_reward(self):
        for speed in (0., 2.):
            self.setUp()
            self.update(.4, settled=False, speed=speed, distance=.5)
            self.update(.46, touching=False)
            self.assertEqual(self.state['release_failure'], 'premature_release')
            _, terms = mobile.reward(self.state, self.launch, parking(.5, False),
                True, False, .46, self.cfg, center=[0,1,0], gravity=[0,0,-2])
            for key in ('valid_launch', 'flight_progress', 'aim', 'basket_hit'):
                self.assertEqual(terms[key], 0.)
            self.assertEqual(terms['task_failure'], -15.)

    def test_release_uses_last_contact_parking_not_detection_parking(self):
        self.unlock()
        self.update(.52)
        # Vehicle starts drifting after release; last-contact eligibility persists.
        self.update(.58, touching=False, settled=False, distance=.3)
        self.assertTrue(self.launch['confirmed'])
        self.assertTrue(self.state['release_qualified'])

    def test_leaving_parking_before_release_is_invalid_even_after_unlock(self):
        self.unlock()
        self.update(.52, settled=False, distance=.3)
        self.update(.58, touching=False)
        self.assertFalse(self.launch.get('confirmed', False))
        self.assertEqual(self.state['release_failure'], 'premature_release')

    def test_progress_is_signed_and_cannot_be_repeated_by_circling(self):
        total = 0.
        for i, distance in enumerate((.3, .1, .3, .1)):
            _, terms = mobile.reward(self.state, self.launch, parking(distance, False),
                False, False, .04*(i+1), self.cfg)
            total += terms['navigation']
        self.assertAlmostEqual(total, 8.*(.5-.1)/.5)

    def test_parking_only_times_out_with_negative_return(self):
        self.unlock()
        value, terms = mobile.reward(self.state, self.launch, parking(), True, False,
            8., self.cfg)
        self.assertLess(value, 0.)
        self.assertEqual(terms['parked'], 3.)
        _, again = mobile.reward(self.state, self.launch, parking(), True, False,
            8., self.cfg)
        self.assertEqual(again['parked'], 0.)
        self.assertEqual(again['task_failure'], 0.)

    def test_throw_reward_preserved_and_collision_outweighs_all_positive_credit(self):
        self.state.update(unlocked=True, release_qualified=True, ready_pending=3.)
        self.launch.update(confirmed=True, launch_pending=5., progress_pending=5.,
                           candidate=dict(position=[0,0,1], velocity=[0,1,0]))
        value, terms = mobile.reward(self.state, self.launch, parking(), True, False,
            1., self.cfg, robot_collision=True, center=[0,1,0], gravity=[0,0,-2], hit=True)
        self.assertEqual(terms['aim'], 10.)
        self.assertEqual(terms['basket_hit'], 20.)
        self.assertEqual(terms['robot_collision'], -60.)
        self.assertLess(value, 0.)

    def test_base_limits_and_initial_stabilization_do_not_modify_input(self):
        requested = np.ones(2)
        np.testing.assert_array_equal(mobile.base_command(self.state, requested, .04, self.cfg), [0.,0.])
        self.state['physics_time'] = .4
        command = mobile.base_command(self.state, requested, .04, self.cfg)
        self.assertAlmostEqual(np.linalg.norm(command), .8*.04)
        for _ in range(100):
            command = mobile.base_command(self.state, requested, .04, self.cfg)
        self.assertAlmostEqual(np.linalg.norm(command), .5)
        np.testing.assert_array_equal(requested, np.ones(2))

    def test_observation_and_autoreset_terminal_metrics(self):
        obs = mobile.observation(self.state, np.zeros(6), np.ones(6), 4., self.cfg)
        self.assertEqual(obs.shape, (16,))
        self.assertEqual(obs[-1], .5)
        records = mobile.terminal_records(dict(success=np.array([False, False]),
            final_info=np.array([dict(success=True), None], dtype=object),
            _final_info=np.array([True, False])), [True, False])
        self.assertEqual(records, [dict(success=True)])

    def test_mobile_checkpoint_cannot_silently_load_fixed_policy(self):
        import ast
        from pathlib import Path
        source = Path(__file__).resolve().parents[1] / 'gym_dcmm/algs/ppo_dcmm/ppo_dcmm_catch_one_stage.py'
        tree = ast.parse(source.read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == '_validate_mobile_checkpoint')
        scope = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), scope)
        policy = N(mobile_basket=True)
        for old in ({}, dict(basket_arm_control='joint_velocity', action_layout='arm6_hand12')):
            with self.assertRaises(ValueError):
                scope['_validate_mobile_checkpoint'](policy, old)
        scope['_validate_mobile_checkpoint'](policy, dict(basket_arm_control='joint_velocity',
            action_layout='base2_arm6_hand12', basket_reward_version='basket_mobile_v1',
            basket_observation_version='mobile51_v1'))

    def test_actual_policy_mobile_action_order_and_scale(self):
        from tests.test_basket_fixed_policy import SCOPE, ArrayTensor
        policy = N(fixed_basket=False, mobile_basket=True, mobile_base_scale=.5,
            action_catch_denorm=[1.5, .025, .15], env=N(call=lambda _: ['Catching']))
        action = np.arange(20, dtype=float).reshape(1, 20)/20.
        result = SCOPE['action2dict'](policy, ArrayTensor(action))
        np.testing.assert_array_equal(result['base'], action[:, :2]*.5)
        np.testing.assert_array_equal(result['arm'], action[:, 2:8]*.025)
        np.testing.assert_array_equal(result['hand'], action[:, 8:20]*.15)

    def test_actual_flight_branch_cannot_bypass_mobile_gate_and_waits_after_hit(self):
        # Use the real environment method without requiring an OpenGL context.
        from tests.test_basket_catching import fixture, scope
        for qualified in (False, True):
            env, park, _ = fixture()
            env.mobile_basket = True
            env.basket_reward_version = 'basket_mobile_v1'
            env.basket_cfg = self.cfg
            self.cfg.basket_floor_z = 0.
            self.cfg.basket_radius = .2
            self.cfg.basket_ball_radius = .04
            env.basket_center = np.array([0., 1., .55])
            env.basket_phase = 'flight'
            env.basket_had_hand_contact = True
            env.start_time = 0.
            env.basket_aim_hit_time = None
            env.basket_mobile_state = mobile.new_state(.5)
            env.basket_mobile_state.update(unlocked=True, release_qualified=qualified,
                                          last_contact_qualified=qualified)
            env.basket_simple_state = dict(confirmed=True,
                candidate=dict(time=.4, position=np.array([0., .4, .6]),
                               direction=np.array([0.,1.]), invalid=False))
            park.update(speed=0., settled=True)
            env.Dcmm.data.time = 1.
            env.Dcmm.data.qpos[37:40] = [0., 1., .54]
            scope['_basket_check_flight'](env, np.array([0., 1., .56]))
            if qualified:
                self.assertEqual(env.basket_aim_hit_time, 1.)
                self.assertFalse(env.terminated)
                env.Dcmm.data.time = 1.31
                scope['_basket_check_flight'](env, env.Dcmm.data.qpos[37:40].copy())
                self.assertEqual(env.terminated_reason, 'basket_score')
            else:
                self.assertIsNone(env.basket_aim_hit_time)
                self.assertEqual(env.terminated_reason, 'not_a_throw')


if __name__ == '__main__':
    unittest.main()
