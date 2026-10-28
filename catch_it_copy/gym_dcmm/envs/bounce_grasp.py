"""Opt-in Bounce Catch control and contact-based holding, isolated from legacy.

The ball's dynamics are supplied by bounce_physical_v2. This module changes
only Catch control/reward/termination and records the version explicitly.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import uuid

import mujoco
import numpy as np

from gym_dcmm.envs.DcmmVecEnv import BounceEnv
from gym_dcmm.utils.roll_rewards import hand_collision_ids

VERSION = 'controlled_hold_v1'
MOVABLE = np.array([0, 2, 3, 4, 6, 7, 8, 10, 11, 13, 14, 15])
FIXED = np.array([1, 5, 9, 12])


@dataclass
class GraspConfig:
    target_step: float = .08
    target_tracking_error: float = .35
    target_margin: float = .02
    torque_limit: float = 1.
    integral_limit: float = .15
    limit_time_constant: float = .002
    hold_seconds: float = .3
    hold_speed: float = .35
    max_joint_violation: float = .08
    abort_joint_violation: float = .25
    min_contact_groups: int = 2
    min_contact_span: float = .15
    contact_force_min: float = .001
    floor_clearance: float = .02
    near_distance: float = .18
    ready_weight: float = .3
    position_weight: float = 2.
    approach_weight: float = 4.
    palm_weight: float = 1.
    velocity_weight: float = 1.
    contact_weight: float = 2.
    wrap_weight: float = 2.
    hold_weight: float = 8.
    success_bonus: float = 25.
    failure_penalty: float = 25.
    step_cost: float = .01
    reward_discount: float = .99
    hand_action_weight: float = .02
    hand_smooth_weight: float = .03
    joint_limit_weight: float = 2.


def configuration(options=None):
    cfg = GraspConfig(**(options or {}))
    for key, value in asdict(cfg).items():
        if not np.isfinite(value) or value < 0:
            raise ValueError('Invalid bounce_grasp option: ' + key)
    for key in ('target_step', 'target_tracking_error', 'torque_limit',
                'limit_time_constant', 'hold_seconds', 'hold_speed',
                'near_distance', 'max_joint_violation'):
        if getattr(cfg, key) <= 0:
            raise ValueError(key + ' must be positive')
    if not 1 <= cfg.min_contact_groups <= 5 or int(cfg.min_contact_groups) != cfg.min_contact_groups:
        raise ValueError('min_contact_groups must be an integer in [1,5]')
    if not 0 <= cfg.min_contact_span <= 1:
        raise ValueError('min_contact_span must be in [0,1]')
    if cfg.abort_joint_violation <= cfg.max_joint_violation:
        raise ValueError('abort_joint_violation must exceed max_joint_violation')
    if not 0 < cfg.reward_discount <= 1:
        raise ValueError('reward_discount must be in (0,1] and match PPO gamma')
    return cfg


def bounded_target(previous, measured, increment, low, high, cfg):
    """Keep requested angles feasible without projecting the physical state."""
    measured = np.clip(measured, low, high)
    previous = np.clip(previous, measured-cfg.target_tracking_error,
                       measured+cfg.target_tracking_error)
    requested = previous + np.clip(increment, -cfg.target_step, cfg.target_step)
    requested = np.clip(requested, measured-cfg.target_tracking_error,
                        measured+cfg.target_tracking_error)
    return np.clip(requested, low, high)


class BoundedHandPID:
    """Hand-only PID: derivative on measurement, conditional integration.

    Torque limit is an explicit simulation prior, not measured motor capacity.
    Existing XML actuator limits still apply, as does post-noise clipping.
    """
    def __init__(self, original, low, high, cfg):
        self.Kp_initial = np.broadcast_to(original.Kp_initial, (16,)).copy()
        self.Ki = np.broadcast_to(original.Ki, (16,)).copy()
        self.Kd = np.broadcast_to(original.Kd, (16,)).copy()
        self.low = np.maximum(np.asarray(low), -cfg.torque_limit)
        self.high = np.minimum(np.asarray(high), cfg.torque_limit)
        self.integral_limit = cfg.integral_limit
        self.reset()

    def reset(self, k=1.):
        self.Kp = self.Kp_initial * k
        self.integral = np.zeros(16)
        self.previous_measurement = None
        self.previous_time = None
        self.saturation = np.zeros(16, dtype=bool)

    def update(self, setpoint, measurement, time):
        measurement = np.asarray(measurement)
        error = np.asarray(setpoint)-measurement
        dt = 0. if self.previous_time is None else max(0., time-self.previous_time)
        derivative = np.zeros(16) if dt <= 0 else (measurement-self.previous_measurement)/dt
        candidate = np.clip(self.integral+self.Ki*error*dt,
                            -self.integral_limit, self.integral_limit)
        raw = self.Kp*error+candidate-self.Kd*derivative
        # Do not integrate farther into saturation.
        blocked = ((raw > self.high) & (error > 0)) | ((raw < self.low) & (error < 0))
        self.integral = np.where(blocked, self.integral, candidate)
        raw = self.Kp*error+self.integral-self.Kd*derivative
        self.saturation = (raw < self.low) | (raw > self.high)
        self.previous_measurement = measurement.copy()
        self.previous_time = float(time)
        return np.clip(raw, self.low, self.high)


def contact_span(directions):
    """A bounded contact spread signal; not a force-closure certificate."""
    if len(directions) < 2:
        return 0.
    directions = np.asarray(directions)
    lengths = np.linalg.norm(directions, axis=1)
    directions = directions[lengths > 1e-9]/lengths[lengths > 1e-9, None]
    return 0. if len(directions) < 2 else float(np.clip((1.-np.min(directions @ directions.T))*.5, 0., 1.))


class BounceGraspEnv(BounceEnv):
    def __init__(self, bounce_grasp=None, **kwargs):
        if kwargs.get('task') != 'Catching' or kwargs.get('object_motion') != 'bounce':
            raise ValueError(VERSION + ' requires Bounce Catching')
        self.grasp_cfg = configuration(bounce_grasp)
        self._grasp_sampling = False
        self._grasp_ready = False
        self._grasp_log_file = None
        self._grasp_token = uuid.uuid4().hex[:12]
        super().__init__(**kwargs)
        self._install_hand_control()
        self._reset_grasp_state()

    def _install_hand_control(self):
        model = self.Dcmm.model
        self._hand_joints = np.array([int(np.flatnonzero(model.jnt_qposadr == 21+i)[0]) for i in range(16)])
        self._hand_ranges = model.jnt_range[self._hand_joints].copy()
        widths = self._hand_ranges[:, 1]-self._hand_ranges[:, 0]
        margin = np.minimum(self.grasp_cfg.target_margin, widths*.1)
        self._target_low = self._hand_ranges[:, 0]+margin
        self._target_high = self._hand_ranges[:, 1]-margin
        self._hand_actuators = np.array([int(np.flatnonzero(model.actuator_trnid[:, 0] == j)[0]) for j in self._hand_joints])
        torque_range = model.actuator_ctrlrange[self._hand_actuators]
        if not isinstance(self.Dcmm.hand_pid, BoundedHandPID):
            self.Dcmm.hand_pid = BoundedHandPID(self.Dcmm.hand_pid, torque_range[:, 0], torque_range[:, 1], self.grasp_cfg)
        # This robot instance only; legacy robots and global PID remain untouched.
        self.Dcmm.action_hand2qpos = self._bounded_hand_action
        self._configure_hand_limits()
        self._palm_id = int(self.hand_start_id)
        self._hand_geoms = set(hand_collision_ids(model, self._palm_id).tolist())
        self._object_geoms = set(np.flatnonzero(model.geom_bodyid == model.body(self.object_name).id).tolist())
        roots = [('mcp_joint', 1), ('mcp_joint_2', 2), ('mcp_joint_3', 3), ('pip_4', 4)]
        root_ids = {model.body(name).id: group for name, group in roots}
        self._contact_groups = {self._palm_id: 0}
        for geom in self._hand_geoms:
            body = int(model.geom_bodyid[geom])
            while body > 0:
                if body in root_ids:
                    self._contact_groups[geom] = root_ids[body]
                    break
                body = int(model.body_parentid[body])
        self._grasp_ready = True

    def _configure_hand_limits(self):
        model = self.Dcmm.model
        # Stiffen only existing hand joint constraints; no qpos corrections.
        model.jnt_solref[self._hand_joints] = [self.grasp_cfg.limit_time_constant, 1.]
        model.jnt_solimp[self._hand_joints] = [.95, .99, .001, .5, 2.]

    def _bounded_hand_action(self, action):
        action = np.asarray(action, dtype=float).reshape(12)
        if not np.isfinite(action).all():
            raise ValueError('Non-finite hand action')
        previous = self.Dcmm.target_hand_qpos.copy()
        measured = self.Dcmm.data.qpos[21:37]
        target = bounded_target(previous[MOVABLE], measured[MOVABLE], action,
                                self._target_low[MOVABLE], self._target_high[MOVABLE], self.grasp_cfg)
        self.Dcmm.target_hand_qpos[MOVABLE] = target
        self.Dcmm.target_hand_qpos[FIXED] = np.clip(0., self._target_low[FIXED], self._target_high[FIXED])
        self._target_clipped = float(np.mean(np.abs(target-(previous[MOVABLE]+action)) > 1e-9))

    def _reset_simulation(self):
        super()._reset_simulation()
        if self._grasp_ready:
            self._configure_hand_limits()

    def _reset_grasp_state(self):
        self._hold_time = 0.
        self._max_hold_time = 0.
        self._was_holding = False
        self._success_latched = False
        self._had_contact = False
        self._target_clipped = 0.
        self._last_sample_time = float(self.Dcmm.data.time)
        self._previous_distance = None
        self._previous_hand_action = np.zeros(12)
        self._reward_sums = {}
        self._episode_ticks = 0
        self._last_grasp_state = self._grasp_state()
        self._reset_step_samples()
        self._previous_potential = float(sum(self._potential_terms(
            self.Dcmm.data.qpos[21+MOVABLE]).values()))

    def _reset_step_samples(self):
        self._sample_count = 0
        self._contact_ticks = 0
        self._capture_ticks = 0
        self._valid_hold_ticks = 0
        self._wrap_score_sum = 0.
        self._span_sum = 0.
        self._groups_sum = 0.
        self._collision_in_step = False
        self._max_violation_in_step = 0.

    def reset(self):
        self._grasp_sampling = False
        obs, info = super().reset()
        self.Dcmm.target_hand_qpos[:] = np.clip(self.Dcmm.target_hand_qpos, self._target_low, self._target_high)
        self._reset_grasp_state()
        info['bounce_grasp_version'] = VERSION
        return obs, info

    def _get_ctrl(self):
        ctrl = super()._get_ctrl()
        # Legacy adds multiplicative actuator noise after PID; enforce total
        # hand limits after that noise as well.
        ctrl[self._hand_actuators] = np.clip(ctrl[self._hand_actuators], self.Dcmm.hand_pid.low, self.Dcmm.hand_pid.high)
        return ctrl

    def _grasp_state(self):
        model, data = self.Dcmm.model, self.Dcmm.data
        rotation = data.body('link6').xmat.reshape(3, 3)
        origin = data.body('link6').xpos
        position = data.body(self.object_name).xpos.copy()
        radius = float(model.geom_size[self.object_id, 0])
        local = rotation.T @ (position-origin)
        # Geometry/FK verification: fingers extend +Y, positive flexion curls
        # toward +Z. The palm-facing normal is +Z, not historical +Y.
        surface = origin + rotation @ np.array([0., .025, .044])
        goal = origin + rotation @ np.array([0., .035, .044+radius])
        jac = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jac, None, position, model.body(self.object_name).id)
        ball_velocity = jac @ data.qvel
        mujoco.mj_jac(model, data, jac, None, goal, model.body('link6').id)
        hand_velocity = jac @ data.qvel
        speed = float(np.linalg.norm(ball_velocity-hand_velocity))
        distance = float(np.linalg.norm(position-goal))
        direction = position-surface
        direction /= max(np.linalg.norm(direction), 1e-9)
        palm_dot = float(np.clip(np.dot(rotation[:, 2], direction), -1., 1.))
        capture = bool(abs(local[0]) <= .075+.2*radius and -.05 <= local[1] <= .15
                       and .044-.5*radius <= local[2] <= .044+2*radius+.07)
        groups, directions = set(), []
        floor_contact = False
        external_support = False
        collision = False
        force = np.zeros(6)
        hand_contact = False
        for index in range(data.ncon):
            contact = data.contact[index]
            a, b = int(contact.geom1), int(contact.geom2)
            if a == self.base_id or b == self.base_id:
                collision = True
            other = b if a in self._object_geoms else a if b in self._object_geoms else None
            if other is None:
                continue
            mujoco.mj_contactForce(model, data, index, force)
            if float(force[0]) < self.grasp_cfg.contact_force_min:
                continue
            if other == self.floor_id:
                floor_contact = True
            if other in self._hand_geoms:
                hand_contact = True
                groups.add(self._contact_groups.get(other, 0))
                directions.append(np.asarray(contact.pos)-position)
            else:
                external_support = True
        q = data.qpos[21:37]
        violation = float(np.max(np.maximum(self._hand_ranges[:, 0]-q, np.maximum(q-self._hand_ranges[:, 1], 0.))))
        target_violation = float(np.max(np.maximum(self._hand_ranges[:, 0]-self.Dcmm.target_hand_qpos,
                                    np.maximum(self.Dcmm.target_hand_qpos-self._hand_ranges[:, 1], 0.))))
        floor_z = float(data.geom_xpos[self.floor_id, 2])
        off_ground = bool(position[2]-radius > floor_z+self.grasp_cfg.floor_clearance and not external_support)
        span = contact_span(directions)
        valid_hold = bool(hand_contact and capture and off_ground and not floor_contact
                          and len(groups) >= self.grasp_cfg.min_contact_groups
                          and span >= self.grasp_cfg.min_contact_span
                          and speed <= self.grasp_cfg.hold_speed
                          and violation <= self.grasp_cfg.max_joint_violation)
        return dict(hand_contact=hand_contact, floor_contact=floor_contact,
                    external_support=external_support, off_ground=off_ground,
                    capture=capture, contact_groups=len(groups), contact_span=span,
                    relative_speed=speed, palm_dot=palm_dot, distance=distance,
                    local_ball=local.tolist(), joint_violation=violation,
                    target_violation=target_violation, valid_hold=valid_hold,
                    collision=collision)

    def _get_contacts(self):
        contacts = super()._get_contacts()
        if self._grasp_sampling and self._grasp_ready:
            state = self._grasp_state()
            time = float(self.Dcmm.data.time)
            dt = max(0., time-self._last_sample_time)
            self._last_sample_time = time
            self._hold_time = (self._hold_time+dt if self._was_holding else 0.) if state['valid_hold'] else 0.
            self._was_holding = state['valid_hold']
            self._max_hold_time = max(self._max_hold_time, self._hold_time)
            self._success_latched = self._success_latched or self._hold_time >= self.grasp_cfg.hold_seconds-1e-9
            self._had_contact = self._had_contact or state['hand_contact']
            self._sample_count += 1
            self._episode_ticks += 1
            self._contact_ticks += int(state['hand_contact'])
            self._capture_ticks += int(state['capture'])
            self._valid_hold_ticks += int(state['valid_hold'])
            self._span_sum += state['contact_span']
            self._groups_sum += state['contact_groups']
            if state['hand_contact'] and state['capture']:
                self._wrap_score_sum += (.5*min(1., state['contact_groups']/3.)
                                         + .5*state['contact_span'])
            self._collision_in_step |= state['collision']
            self._max_violation_in_step = max(self._max_violation_in_step, state['joint_violation'])
            self._last_grasp_state = state
        return contacts

    def _potential_terms(self, q):
        """Bounded state scores, paid only through discounted differences.

        Paying these scores every step would reward loitering or repeated
        touches more than finishing a short successful hold.
        """
        cfg, state = self.grasp_cfg, self._last_grasp_state
        distance = state['distance']
        near = float(np.exp(-(distance/cfg.near_distance)**2))
        count = max(1, self._sample_count)
        contact_fraction = self._contact_ticks/count
        ready = np.tile([.15, .1, .1], 4)
        ready[9:] = [.2, .2, .1]
        shape_score = float(np.exp(-np.mean(((q-ready)/.5)**2)))
        return dict(
            position=cfg.position_weight/(1.+(distance/.35)**2),
            precision=cfg.approach_weight/(1.+(distance/cfg.near_distance)**2),
            palm=cfg.palm_weight*near*.25*(state['palm_dot']+1.)**2,
            velocity=cfg.velocity_weight*near/(1.+(state['relative_speed']/.6)**2),
            ready=cfg.ready_weight*near*(1.-contact_fraction)*shape_score,
            contact=cfg.contact_weight*contact_fraction,
            wrap=cfg.wrap_weight*self._wrap_score_sum/count,
            hold=cfg.hold_weight*min(1., self._hold_time/cfg.hold_seconds))

    def compute_reward(self, obs, info, ctrl):
        cfg = self.grasp_cfg
        if np.asarray(obs['hand']).shape != (12,):
            raise ValueError('Expected the original 12 movable hand angles')
        # Reward uses true simulation angles; policy observations retain their
        # original noise and dimensionality.
        scores = self._potential_terms(self.Dcmm.data.qpos[21+MOVABLE])
        current = float(sum(scores.values()))
        terminal = bool(info['bounce_grasp_terminal'])
        next_potential = 0. if terminal else current
        previous = self._previous_potential
        hand_action = np.asarray(ctrl['hand'], dtype=float).reshape(12)
        terms = dict(
            shaping=cfg.reward_discount*next_potential-previous,
            step_cost=-cfg.step_cost,
            hand_action=-cfg.hand_action_weight*float(np.mean((hand_action/cfg.target_step)**2)),
            hand_smooth=-cfg.hand_smooth_weight*float(np.mean(((hand_action-self._previous_hand_action)/cfg.target_step)**2)),
            joint_limit=-cfg.joint_limit_weight*min(10., (self._max_violation_in_step/cfg.max_joint_violation)**2),
            robot_control=-.02*float(np.linalg.norm(ctrl['base']))-.02*float(np.linalg.norm(np.asarray(ctrl['arm'])/.025)),
            arm_limit=0. if self.arm_limit else -.3,
            collision=-10. if self._collision_in_step else 0.,
            success=cfg.success_bonus if info['success'] else 0.,
            failure=-cfg.failure_penalty if terminal and not info['success'] else 0.)
        self._previous_potential = next_potential
        self._previous_hand_action = hand_action.copy()
        info['bounce_grasp_reward_terms'] = terms
        info['bounce_grasp_potential_terms'] = scores
        info['bounce_grasp_potential_previous'] = previous
        info['bounce_grasp_potential'] = next_potential
        for name, value in terms.items():
            self._reward_sums[name] = self._reward_sums.get(name, 0.)+float(value)
        return float(sum(terms.values()))

    def step(self, action):
        self.steps += 1
        self._reset_step_samples()
        self._last_sample_time = float(self.Dcmm.data.time)
        self._grasp_sampling = True
        try:
            # Keep original robot dynamics, IK, action delays, controller ticks,
            # collision/out-of-bounds checks; do not invoke legacy step's .25m
            # proximity stage/termination/success contract.
            self._step_mujoco_simulation(action)
        finally:
            self._grasp_sampling = False
        obs = self._get_obs()
        info = self._get_info()
        finite = bool(np.isfinite(self.Dcmm.data.qpos).all() and np.isfinite(self.Dcmm.data.qvel).all())
        if self._collision_in_step:
            self.terminated = True
            self.terminated_reason = 'base_collision'
        elif not finite:
            self.terminated = True
            self.terminated_reason = 'nonfinite_state'
        elif self._max_violation_in_step > self.grasp_cfg.abort_joint_violation:
            self.terminated = True
            self.terminated_reason = 'hand_joint_limit'
        elif self.terminated:
            self.terminated_reason = self.terminated_reason or 'out_of_bounds'
        success = bool(self._hold_time >= self.grasp_cfg.hold_seconds-1e-9
                       and not self.terminated and self._last_grasp_state['valid_hold'])
        if success:
            self.terminated = True
            self.terminated_reason = 'controlled_hold'
        self.stage = 'grasping' if self._last_grasp_state['hand_contact'] else 'tracking'
        truncated = bool(info['env_time'] >= self.env_time)
        info.update(success=success, success_version='bounce_controlled_hold_v1',
                    bounce_grasp_terminal=bool(self.terminated or truncated),
                    bounce_grasp_version=VERSION, terminated_reason=self.terminated_reason or ('timeout' if truncated else None),
                    bounce_grasp_hold_seconds=float(self._hold_time),
                    bounce_grasp_max_hold_seconds=float(self._max_hold_time),
                    bounce_grasp_contact_fraction=self._contact_ticks/max(1, self._sample_count),
                    bounce_grasp_had_contact=self._had_contact,
                    bounce_grasp_hand_target_clipped=self._target_clipped,
                    bounce_grasp_state=dict(self._last_grasp_state),
                    bounce_grasp_joint_violation=self._max_violation_in_step,
                    bounce_grasp_target_violation=self._last_grasp_state['target_violation'])
        reward = self.compute_reward(obs, info, action)
        self.info.update(env_time=info['env_time'], ee_distance=info['ee_distance'], base_distance=info['base_distance'])
        info['imgs'] = self.render()
        info['ctrl_params'] = np.concatenate((self.k_arm, self.k_drive, self.k_hand,
            [len(self.action_buffer[k]) for k in ('base', 'arm', 'hand')]))
        if self.terminated or truncated:
            summary = dict(version=VERSION, episode=self.episode_count, success=success,
                reason=info['terminated_reason'], physics=getattr(self, 'bounce_v2_episode', {}),
                policy_steps=self.steps, control_ticks=self._episode_ticks,
                max_hold_seconds=self._max_hold_time, sums=self._reward_sums.copy())
            info['bounce_grasp_episode'] = summary
            if self._grasp_log_file:
                with self._grasp_log_file.open('a') as file:
                    file.write(json.dumps(summary)+'\n')
        return obs, reward, bool(self.terminated), truncated, info

    def set_bounce_grasp_log_dir(self, directory):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self._grasp_log_file = path/('episodes_{}_{}.jsonl'.format(os.getpid(), self._grasp_token))


def write_provenance(output, config):
    root = Path(__file__).resolve().parents[2]
    paths = ['gym_dcmm/envs/bounce_grasp.py', 'gym_dcmm/envs/bounce_physical_v2.py',
             'gym_dcmm/envs/DcmmVecEnv.py', 'gym_dcmm/agents/MujocoDcmm_bounce_5fe75d5f.py',
             'gym_dcmm/utils/pid.py', 'train_DCMM.py', 'configs/config.yaml',
             'configs/train/DcmmPPO.yaml', 'configs/env/DcmmCfg_bounce_5fe75d5f.py',
             'assets/urdf/x1_xarm6_leap_right_object.xml',
             'gym_dcmm/algs/ppo_dcmm_5fe75d5f/ppo_dcmm_catch_hold.py',
             'gym_dcmm/algs/ppo_dcmm_5fe75d5f/ppo_dcmm_catch_two_stage.py',
             'gym_dcmm/algs/ppo_dcmm_5fe75d5f/models_catch.py',
             'scripts/train_bounce_three_ball_hold_v1.sh']
    checkpoints = {}
    for key in ('checkpoint_tracking', 'checkpoint_catching'):
        if config.get(key):
            path = Path(config[key]).resolve()
            checkpoints[key] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    result = dict(version=VERSION, options=asdict(configuration(config.get('bounce_grasp'))),
        training_options=config.get('bounce_grasp_training', {}), checkpoints=checkpoints,
        baseline_manifest='backups/bounce_before_controlled_hold_20261028_195129/manifest.json',
        success='continuous real multi-region hand contact, capture geometry, low relative speed, no external support, valid joints; not proximity',
        notes='Hand torque/limit settings are engineering simulation priors, not real-hardware calibration. Continuous hold sampled at 2ms control ticks, not every 125us physics substep.',
        sources={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths})
    (Path(output)/'bounce_grasp_provenance.json').write_text(json.dumps(result, indent=2))
