"""Physical navigate--settle--throw task, isolated from fixed-base rewards."""
import numpy as np

from gym_dcmm.utils.basket_aim import make_config as aim_config, reward as aim_reward
from gym_dcmm.utils.basket_simple import update_launch


DEFAULTS = dict(target_x_range=[-.4, .4], target_y_range=[1.35, 1.85],
                max_time=8., max_speed=.5, max_acceleration=.8,
                position_tolerance=.10, speed_tolerance=.10, settle_seconds=.20,
                carry_settle_seconds=.30)


def make_config(base, options=None):
    options = dict(options or {})
    unknown = set(options) - set(DEFAULTS)
    if unknown:
        raise ValueError('Unknown basket_mobile options: ' + str(sorted(unknown)))
    values = dict(DEFAULTS, **options)
    for key in ('target_x_range', 'target_y_range'):
        pair = np.asarray(values[key], dtype=float)
        if pair.shape != (2,) or not np.all(np.isfinite(pair)) or pair[0] > pair[1]:
            raise ValueError(key + ' must be a finite ordered pair')
        values[key] = pair.copy()
    if values['target_y_range'][0] < 1.05:
        raise ValueError('Mobile target_y_range must stay >= 1.05m')
    for key in set(DEFAULTS) - {'target_x_range', 'target_y_range'}:
        if not np.isfinite(float(values[key])) or float(values[key]) <= 0:
            raise ValueError(key + ' must be finite and positive')
    cfg = aim_config(base)
    for key, value in values.items():
        setattr(cfg, 'mobile_' + key, value)
    cfg.basket_track_forward_offset = None
    cfg.basket_track_position_tolerance = values['position_tolerance']
    cfg.basket_track_speed_tolerance = values['speed_tolerance']
    cfg.basket_track_max_speed = values['max_speed']
    cfg.basket_track_slowdown_gain = 1.5
    cfg.basket_max_time = values['max_time']
    cfg.basket_simple_collision_cost = 60.
    cfg.basket_fail_dist = max(float(base.basket_fail_dist),
                              float(np.hypot(max(abs(values['target_x_range'])),
                                             values['target_y_range'][1])) + 1.)
    return cfg


def sample_target(cfg, rng):
    return np.array([rng.uniform(*cfg.mobile_target_x_range),
                     rng.uniform(*cfg.mobile_target_y_range), cfg.basket_center[2]])


def new_state(distance):
    return dict(unlocked=False, settle_seconds=0., initial_distance=float(distance),
                previous_distance=float(distance), ready_pending=0., touching=False,
                release_qualified=False, release_failure=None,
                previous_command=np.zeros(2), elapsed_reward_time=0., physics_time=0.)


def update_physics(state, launch, parking, position, velocity, center,
                   now, dt, touching, collision, cfg):
    """Capture eligibility at LAST contact, never at delayed release detection."""
    state['touching'] = bool(touching)
    state['physics_time'] = float(now)
    if not state['unlocked']:
        state['settle_seconds'] = (state['settle_seconds'] + dt
                                   if now >= cfg.mobile_carry_settle_seconds and
                                   parking['settled'] and touching and not collision else 0.)
        if state['settle_seconds'] + 1e-9 >= cfg.mobile_settle_seconds:
            state['unlocked'] = True
            state['ready_pending'] = 3.
            state['ready_time'] = float(now)
    if touching and not launch.get('confirmed'):
        state['last_contact_qualified'] = bool(state['unlocked'] and parking['settled'])
        state['release_parking_distance'] = parking['distance']
        state['release_base_speed'] = parking['speed']
    candidate = launch.get('candidate')
    # The reset ball settles onto the cup under gravity. During this initial
    # interval the base remains still; contact flicker is not a policy release.
    if not touching and now < cfg.mobile_carry_settle_seconds:
        return
    # Includes slow drops that would never pass launch's forward-speed threshold.
    if (not touching and candidate is not None and not launch.get('confirmed')
            and now-candidate['time'] >= cfg.basket_simple_free_seconds
            and not state.get('last_contact_qualified', False)):
        state['release_failure'] = 'premature_release'
        launch['stopped'] = True
        return
    if (not touching and candidate is None and now >= cfg.mobile_carry_settle_seconds):
        state['release_failure'] = 'ball_not_held'
        return
    update_launch(launch, position, velocity, center, now, touching, collision, cfg)
    if launch.get('confirmed'):
        state['release_qualified'] = bool(state.get('last_contact_qualified', False))


def base_command(state, requested, dt, cfg):
    """Policy steering with speed/acceleration limits; no automatic navigation."""
    requested = np.array(requested, dtype=float, copy=True)
    if state['physics_time'] < cfg.mobile_carry_settle_seconds:
        requested[:] = 0.
    requested *= min(1., cfg.mobile_max_speed / max(float(np.linalg.norm(requested)), 1e-9))
    previous = state['previous_command']
    delta = requested - previous
    delta *= min(1., cfg.mobile_max_acceleration*dt / max(float(np.linalg.norm(delta)), 1e-9))
    command = previous + delta
    state['previous_command'] = command.copy()
    return command


def observation(state, joints, joint_velocity, elapsed, cfg):
    return np.r_[joints, joint_velocity, float(state['unlocked']),
                 min(state['settle_seconds']/cfg.mobile_settle_seconds, 1.),
                 float(state['touching']), np.clip(1.-elapsed/cfg.basket_max_time, 0., 1.)]


def reward(state, launch, parking, done, success, elapsed, cfg, *,
           robot_collision=False, arm_action=None, base_action=None,
           center=None, gravity=None, hit=False):
    # Invalid/early releases cannot receive launch, flight, aim or hit credit.
    value, terms = aim_reward(launch, done, cfg, robot_collision=robot_collision,
                             arm_action=arm_action if state['unlocked'] else None,
                             center=center, gravity=gravity,
                             hit=hit and state['release_qualified'])
    dt = max(0., elapsed-state['elapsed_reward_time'])
    state['elapsed_reward_time'] = float(elapsed)
    terms.update(navigation=0., navigation_distance=0., navigation_velocity=0.,
                 parked=state.pop('ready_pending', 0.), braking=0.,
                 base_control=0., base_smoothness=0., time=-.15*dt, task_failure=0.)
    if not state['unlocked']:
        initial = max(state['initial_distance'], .25)
        old = np.clip(state['previous_distance'], 0., initial)
        current = np.clip(parking['distance'], 0., initial)
        terms['navigation'] = 8.*float(old-current)/initial
        terms['navigation_distance'] = -.5*parking['distance']*dt
        terms['navigation_velocity'] = -.5*float(np.sum(
            (parking['velocity']-parking['desired_velocity'])**2))*dt
    else:
        # Still applies in flight: driving away after unlocking gives no benefit.
        terms['braking'] = -2.*(parking['speed']**2 + parking['distance']**2)*dt
        if not state.get('navigation_finished'):
            initial = max(state['initial_distance'], .25)
            terms['navigation'] = 8.*float(np.clip(state['previous_distance'], 0., initial)
                - np.clip(parking['distance'], 0., initial))/initial
            state['navigation_finished'] = True
    state['previous_distance'] = parking['distance']
    if base_action is not None:
        action = np.asarray(base_action, dtype=float)
        terms['base_control'] = -.02*float(np.mean(action**2))*dt
        previous = state.get('previous_base_action')
        if previous is not None:
            terms['base_smoothness'] = -.02*float(np.mean((action-previous)**2))
        state['previous_base_action'] = action.copy()
    if done and not success and not state.get('failure_paid'):
        terms['task_failure'] = -15.
        state['failure_paid'] = True
    return float(sum(terms.values())), terms


def terminal_records(infos, dones):
    """Vector autoreset replaces top-level fields with the NEXT episode's info."""
    records = []
    for index in np.flatnonzero(dones):
        final = infos.get('final_info')
        mask = infos.get('_final_info')
        if final is not None and (mask is None or mask[index]) and isinstance(final[index], dict):
            records.append(final[index])
        else:
            records.append({key: value[index] for key, value in infos.items()
                            if not key.startswith('_') and isinstance(value, np.ndarray)
                            and value.ndim > 0})
    return records
