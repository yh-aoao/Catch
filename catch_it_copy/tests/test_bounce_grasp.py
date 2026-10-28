"""Behavioral checks for opt-in Bounce Catch holding and bounded control."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from gym_dcmm.envs.bounce_grasp import (BoundedHandPID, BounceGraspEnv,
    MOVABLE, bounded_target, configuration, contact_span)
from gym_dcmm.envs.DcmmVecEnv import BounceEnv
from gym_dcmm.envs.bounce_physical_v2 import make_bounce_physical_v2


class GraspMathTests(unittest.TestCase):
    def test_target_does_not_accumulate_outside_limits_or_far_from_state(self):
        cfg = configuration()
        target = np.zeros(12)
        for _ in range(200):
            target = bounded_target(target, np.zeros(12), np.ones(12),
                                    -.2*np.ones(12), 2*np.ones(12), cfg)
        self.assertLessEqual(float(target.max()), cfg.target_tracking_error+1e-9)
        for _ in range(200):
            target = bounded_target(target, -.1*np.ones(12), -np.ones(12),
                                    -.2*np.ones(12), 2*np.ones(12), cfg)
        self.assertGreaterEqual(float(target.min()), -.2)

    def test_pid_output_is_bounded_and_integrator_does_not_wind_up(self):
        original = SimpleNamespace(Kp_initial=np.ones(16), Ki=1., Kd=np.ones(16))
        pid = BoundedHandPID(original, -2*np.ones(16), 2*np.ones(16), configuration())
        for i in range(200):
            output = pid.update(np.ones(16)*100, np.zeros(16), i*.002)
            self.assertTrue(np.all(np.abs(output) <= 1.))
        np.testing.assert_array_equal(pid.integral, 0)
        # Changing only the target must not create a derivative kick.
        pid.reset()
        pid.Kp[:] = 0
        pid.Ki[:] = 0
        pid.update(np.zeros(16), np.zeros(16), 0.)
        np.testing.assert_array_equal(pid.update(np.ones(16), np.zeros(16), .002), 0)

    def test_contact_spread_distinguishes_same_side_and_opposing(self):
        self.assertEqual(contact_span([]), 0.)
        self.assertEqual(contact_span([[1, 0, 0], [1, 0, 0]]), 0.)
        self.assertEqual(contact_span([[1, 0, 0], [-1, 0, 0]]), 1.)

    def test_hold_duration_resets_on_lost_contact_at_controller_ticks(self):
        env = object.__new__(BounceGraspEnv)
        env.grasp_cfg = configuration()
        env.Dcmm = SimpleNamespace(data=SimpleNamespace(time=0.))
        env._grasp_sampling = env._grasp_ready = True
        env._hold_time = env._max_hold_time = 0.
        env._was_holding = env._success_latched = env._had_contact = False
        env._last_sample_time = 0.
        env._episode_ticks = 0
        env._reset_step_samples()
        state = dict(valid_hold=True, hand_contact=True, capture=True,
                     contact_span=1., contact_groups=2, collision=False, joint_violation=0.)
        env._grasp_state = lambda: state.copy()
        with patch.object(BounceEnv, '_get_contacts', return_value={}):
            for tick in range(1, 101):
                env.Dcmm.data.time = tick*.002
                env._get_contacts()
            self.assertAlmostEqual(env._hold_time, .198)
            self.assertFalse(env._success_latched)
            state['valid_hold'] = False
            env.Dcmm.data.time += .002
            env._get_contacts()
            self.assertEqual(env._hold_time, 0.)
            state['valid_hold'] = True
            for _ in range(152):
                env.Dcmm.data.time += .002
                env._get_contacts()
            self.assertGreaterEqual(env._hold_time, .3)
            self.assertTrue(env._success_latched)


class GraspRuntimeTests(unittest.TestCase):
    material = Path(__file__).resolve().parents[1]/'ycb_objects/calibration_data/balls.three_ball_training_v1.json'

    def make(self, **options):
        return make_bounce_physical_v2(task='Catching', object_motion='bounce',
            viewer=False, render_mode=None, bounce_material_file=self.material,
            bounce_launch='baseball_intercept_v1', **options)

    def test_legacy_and_tracking_routes_remain_separate(self):
        legacy = self.make()
        try:
            self.assertNotIsInstance(legacy, BounceGraspEnv)
            original_pid = legacy.Dcmm.hand_pid.__class__
            self.assertNotEqual(original_pid, BoundedHandPID)
        finally:
            legacy.close()
        with self.assertRaises(ValueError):
            make_bounce_physical_v2(task='Tracking', object_motion='bounce',
                bounce_grasp_version='controlled_hold_v1', viewer=False, render_mode=None)

    def test_three_profiles_reset_and_force_extreme_actions(self):
        for ball in ('default', 'ycb:tennis_ball', 'ycb:baseball'):
            env = self.make(test_object=ball, bounce_grasp_version='controlled_hold_v1')
            try:
                for seed in (0, 1):
                    obs, info = env.reset(seed=seed)
                    self.assertEqual(obs['hand'].shape, (12,))
                    self.assertEqual(env.obs_c_dim, 30)
                    self.assertEqual(info['bounce_grasp_version'], 'controlled_hold_v1')
                    for direction in (-1., 1.):
                        for _ in range(8):
                            obs, reward, terminated, truncated, info = env.step(dict(
                                base=np.zeros(2), arm=np.zeros(6), hand=np.full(12, direction*.15)))
                            self.assertTrue(np.isfinite(reward))
                            self.assertEqual(info['bounce_grasp_target_violation'], 0.)
                            self.assertLess(info['bounce_grasp_joint_violation'], .03)
                            self.assertTrue(np.all(np.abs(env.Dcmm.data.ctrl[env._hand_actuators]) <= 1.+1e-9))
                            self.assertFalse(info['success'])
                            if terminated or truncated:
                                env.reset(seed=seed)
            finally:
                env.close()

    def test_ball_physics_and_launch_preserved(self):
        for ball in ('default', 'ycb:tennis_ball', 'ycb:baseball'):
            values = []
            for version in ('legacy', 'controlled_hold_v1'):
                env = self.make(test_object=ball, bounce_grasp_version=version)
                try:
                    env.reset(seed=123)
                    pair = env.Dcmm.model.pair('bounce_v2_ground').id
                    values.append((env.object_pos3d.copy(), env.object_vel6d.copy(),
                        env.Dcmm.model.pair_solref[pair].copy(), env.Dcmm.model.pair_friction[pair].copy(),
                        env.bounce_v2_episode.copy()))
                finally:
                    env.close()
            for a, b in zip(values[0][:-1], values[1][:-1]):
                np.testing.assert_array_equal(a, b)
            self.assertEqual(values[0][-1], values[1][-1])

    def test_mixed_factory_and_timeout_are_not_proximity_success(self):
        env = self.make(bounce_grasp_version='controlled_hold_v1', bounce_ball_mode='mixed',
            bounce_ball_weights=dict(original=.6, tennis_ball=.2, baseball=.2))
        try:
            profiles = set()
            for seed in range(8):
                _, info = env.reset(seed=seed)
                profiles.add(info['bounce_profile'])
            self.assertGreater(len(profiles), 1)
            # Place a ball within the old .25m test, behind the palm and with no
            # true holding contacts. A timeout must still be a failure.
            env.env_time = .04
            body = env.Dcmm.data.body('link6')
            position = body.xpos-body.xmat.reshape(3, 3)[:, 2]*.2
            env.Dcmm.set_throw_pos_vel(pose=np.r_[position, [1, 0, 0, 0]], velocity=np.zeros(6))
            env.object_throw = True
            mujoco.mj_forward(env.Dcmm.model, env.Dcmm.data)
            _, reward, terminated, truncated, info = env.step(dict(
                base=np.zeros(2), arm=np.zeros(6), hand=np.zeros(12)))
            self.assertTrue(terminated or truncated)
            self.assertFalse(info['success'])
            self.assertEqual(info['success_version'], 'bounce_controlled_hold_v1')
        finally:
            env.close()


class GraspStepContractTests(unittest.TestCase):
    """Controlled tick fixtures test the contract, not dynamic grasp ability."""

    def make(self):
        return make_bounce_physical_v2(task='Catching', object_motion='bounce',
            viewer=False, render_mode=None,
            bounce_material_file=GraspRuntimeTests.material,
            bounce_launch='baseball_intercept_v1',
            bounce_grasp_version='controlled_hold_v1')

    def fixture(self, env, state_at_tick):
        state = env._grasp_state()
        state.update(valid_hold=True, hand_contact=True, capture=True,
                     off_ground=True, floor_contact=False, external_support=False,
                     contact_groups=2, contact_span=1., relative_speed=0.,
                     joint_violation=0., target_violation=0., collision=False)
        current = [state]
        ticks = [0]

        def advance(action):
            for _ in range(env.steps_per_policy):
                ticks[0] += 1
                current[0] = dict(state, **state_at_tick(ticks[0]))
                env.Dcmm.data.time += .002
                env._get_contacts()

        return advance, lambda: current[0].copy()

    def action(self):
        return dict(base=np.zeros(2), arm=np.zeros(6), hand=np.zeros(12))

    def test_step_accepts_completed_hold_and_emits_success_reward(self):
        env = self.make()
        try:
            env.reset(seed=0)
            advance, state = self.fixture(env, lambda tick: {})
            with patch.object(env, '_step_mujoco_simulation', side_effect=advance), \
                    patch.object(env, '_grasp_state', side_effect=state):
                for _ in range(7):
                    _, _, terminated, truncated, info = env.step(self.action())
                    self.assertFalse(terminated or truncated)
                    self.assertFalse(info['success'])
                _, reward, terminated, truncated, info = env.step(self.action())
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertTrue(info['success'])
            self.assertEqual(info['terminated_reason'], 'controlled_hold')
            self.assertGreaterEqual(info['bounce_grasp_hold_seconds'], .3)
            self.assertEqual(info['bounce_grasp_reward_terms']['success'],
                             env.grasp_cfg.success_bonus)
            self.assertTrue(np.isfinite(reward))
        finally:
            env.close()

    def test_lost_hold_at_last_tick_rejects_success_and_restarts_duration(self):
        env = self.make()
        try:
            env.reset(seed=0)
            # Reach .3s within step 8, then lose contact on its final tick.
            advance, state = self.fixture(env, lambda tick: dict(
                valid_hold=False, hand_contact=False) if tick == 160 else {})
            with patch.object(env, '_step_mujoco_simulation', side_effect=advance), \
                    patch.object(env, '_grasp_state', side_effect=state):
                for _ in range(8):
                    _, _, terminated, truncated, info = env.step(self.action())
                self.assertFalse(terminated or truncated)
                self.assertFalse(info['success'])
                self.assertEqual(info['bounce_grasp_hold_seconds'], 0.)
                self.assertGreaterEqual(info['bounce_grasp_max_hold_seconds'], .3)
                for _ in range(7):
                    _, _, terminated, truncated, info = env.step(self.action())
                    self.assertFalse(info['success'])
                _, _, terminated, _, info = env.step(self.action())
            self.assertTrue(terminated)
            self.assertTrue(info['success'])
        finally:
            env.close()

    def test_collision_and_joint_violation_override_a_completed_hold(self):
        for overrides, expected in [
                (dict(collision=True), 'base_collision'),
                (dict(joint_violation=.3), 'hand_joint_limit')]:
            with self.subTest(reason=expected):
                env = self.make()
                try:
                    env.reset(seed=0)
                    advance, state = self.fixture(env, lambda tick: overrides if tick == 160 else {})
                    with patch.object(env, '_step_mujoco_simulation', side_effect=advance), \
                            patch.object(env, '_grasp_state', side_effect=state):
                        for _ in range(8):
                            _, _, terminated, _, info = env.step(self.action())
                    self.assertTrue(terminated)
                    self.assertFalse(info['success'])
                    self.assertEqual(info['terminated_reason'], expected)
                    self.assertEqual(info['bounce_grasp_reward_terms']['success'], 0.)
                finally:
                    env.close()

    def test_wrist_or_floor_support_rejects_an_otherwise_valid_capture(self):
        env = self.make()
        try:
            env.reset(seed=0)
            data = env.Dcmm.data
            frame = data.body('link6')
            ball = frame.xpos + frame.xmat.reshape(3, 3) @ np.array([0., .035, .084])
            obj_geom = next(iter(env._object_geoms))
            hand = sorted(env._hand_geoms - {env._palm_id})[0]
            contacts = [SimpleNamespace(geom1=obj_geom, geom2=env._palm_id,
                                       pos=ball+np.array([.04, 0., 0.])),
                        SimpleNamespace(geom1=obj_geom, geom2=hand,
                                       pos=ball-np.array([.04, 0., 0.]))]
            proxy = SimpleNamespace(ncon=2, contact=contacts, qpos=data.qpos,
                                    qvel=data.qvel, geom_xpos=data.geom_xpos)
            proxy.body = lambda name: SimpleNamespace(xpos=ball) if name == env.object_name else data.body(name)

            def force(model, target_data, index, output):
                output[:] = 0.
                output[0] = 1.

            with patch.object(env.Dcmm, 'data', proxy), \
                    patch('gym_dcmm.envs.bounce_grasp.mujoco.mj_jac'), \
                    patch('gym_dcmm.envs.bounce_grasp.mujoco.mj_contactForce', side_effect=force):
                self.assertTrue(env._grasp_state()['valid_hold'])
                # Geom62 is the wrist's link6 mesh, intentionally excluded
                # from the palm/finger hand set. Resolve it by mesh/body ID.
                model = env.Dcmm.model
                candidates = [g for g in range(model.ngeom)
                    if int(model.geom_bodyid[g]) == model.body('link6').id
                    and (model.geom_contype[g] or model.geom_conaffinity[g])
                    and g not in env._hand_geoms]
                self.assertTrue(candidates)
                for support in [candidates[0], env.floor_id]:
                    proxy.contact = contacts + [SimpleNamespace(geom1=obj_geom,
                        geom2=support, pos=ball+np.array([0., 0., -.04]))]
                    proxy.ncon = 3
                    state = env._grasp_state()
                    self.assertTrue(state['external_support'])
                    self.assertFalse(state['off_ground'])
                    self.assertFalse(state['valid_hold'])
        finally:
            env.close()

    def test_real_fk_contact_geometry_can_satisfy_capture_predicate(self):
        """A static contact fixture establishes feasibility, not .3s holding."""
        env = self.make()
        try:
            env.reset(seed=0)
            model, data = env.Dcmm.model, env.Dcmm.data
            data.qvel[:] = 0.
            data.qpos[21+MOVABLE] = np.r_[np.tile([1., .75, .65], 3), [1.2, 1.2, 1.2]]
            mujoco.mj_forward(model, data)
            palm = data.body('link6')
            point = palm.xpos + palm.xmat.reshape(3, 3) @ np.array([0., 0., .11])
            env.Dcmm.set_throw_pos_vel(pose=np.r_[point, [1., 0., 0., 0.]],
                                      velocity=np.zeros(6))
            mujoco.mj_forward(model, data)
            state = env._grasp_state()
            self.assertTrue(state['capture'])
            self.assertTrue(state['hand_contact'])
            self.assertGreaterEqual(state['contact_groups'], 2)
            self.assertGreaterEqual(state['contact_span'], env.grasp_cfg.min_contact_span)
            self.assertFalse(state['external_support'])
            self.assertTrue(state['valid_hold'])
            # No controller ticks or elapsed holding were fabricated by the
            # geometry setup, so this must not be recorded as task success.
            self.assertEqual(env._hold_time, 0.)
            self.assertFalse(env._success_latched)
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
