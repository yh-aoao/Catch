"""World-coordinate basket events and phase-specific throwing rewards."""
import numpy as np


def joint_throw_reward(task, phase, velocity, reference, holding, release_quality,
                       distance, previous_distance, controls, success, failed,
                       dt, state, cfg, launch_quality=None):
    """Episode-bounded preparation credit; release and score are events.

    state belongs to one environment and is cleared on reset. No ball actuation.
    """
    terms = dict(time=-cfg.basket_catch_time_cost, support=0.,
                 velocity_progress=0., valid_release=0., flight_progress=0.,
                 control=0., smoothness=0., success=0., failure=0.)
    if phase == 'preparing' and holding:
        # Small total support budget, no early hold penalty during locomotion.
        duration = min(dt, max(0., cfg.basket_support_budget_seconds - state.get('support', 0.)))
        terms['support'] = cfg.basket_support_reward_rate * duration
        state['support'] = state.get('support', 0.) + duration
        # Credit only new episode-best ballistic quality; recontacts cannot farm it.
        readiness = float(np.clip(launch_quality or 0., 0., 1.))
        if 'best_readiness' not in state:
            state['best_readiness'] = readiness  # no credit for the reset pose
        best = state['best_readiness']
        terms['velocity_progress'] = cfg.basket_w_velocity_progress * max(0., readiness-best)
        state['best_readiness'] = max(best, readiness)
    quality = float(np.clip(release_quality, 0., 1.))
    best_release = state.get('best_release', 0.)
    terms['valid_release'] = cfg.basket_w_valid_release * max(0., quality-best_release)
    state['best_release'] = max(best_release, quality)
    # No distance-to-basket reward: carrying/recontact must not earn flight credit.
    # Track has no hand action cost or hand-pose credit.
    keys = ('base', 'arm', 'hand') if task == 'Catching' else ('base', 'arm')
    for key in keys:
        action = np.asarray(controls.get(key, []), dtype=float)
        if not action.size:
            continue
        weight = cfg.basket_throw_ctrl_hand if key == 'hand' else cfg.basket_throw_ctrl_arm
        terms['control'] -= weight * float(np.mean(action**2))
        old = state.get('action_'+key)
        if old is not None:
            terms['smoothness'] -= cfg.basket_joint_smooth_weight * float(np.mean((action-old)**2))
        state['action_'+key] = action.copy()
    if not state.get('terminal', False):
        terms['success'] = cfg.basket_w_score if success else 0.
        terms['failure'] = -cfg.basket_catch_failure_cost if failed and not success else 0.
        state['terminal'] = bool(success or failed)
    return float(sum(terms.values())), terms


def hoop_crossing(previous, current, center, tilt_deg, hoop_radius, ball_radius):
    normal = np.array([0., -np.sin(np.radians(tilt_deg)), np.cos(np.radians(tilt_deg))])
    previous, current, center = map(np.asarray, (previous, current, center))
    d0, d1 = np.dot(previous - center, normal), np.dot(current - center, normal)
    crossed = bool(d0 > 0. and d1 <= 0.)
    if not crossed:
        return False, False, None
    point = previous + (d0 / (d0 - d1)) * (current - previous)
    radial = float(np.linalg.norm(point - center))
    return True, bool(radial <= max(0., hoop_radius - ball_radius)), radial


def flight_failure(position, center, radius, floor_contact, cfg):
    if floor_contact or position[2] <= cfg.basket_floor_z + radius:
        return 'ball_on_floor'
    if np.linalg.norm(np.asarray(position) - center) > cfg.basket_fail_dist:
        return 'ball_too_far'
    return None


def predicted_miss(position, velocity, center, gravity):
    # Dense preparation signal even when a launch cannot yet reach hoop height.
    times = np.linspace(.04, 1.5, 40)
    path = np.asarray(position) + times[:, None] * velocity
    path += .5 * times[:, None] ** 2 * np.asarray(gravity)
    return float(np.min(np.linalg.norm(path - center, axis=1)))


def catching_reward(phase, parking, previous_parking_distance, distance,
                    previous_flight_distance, predicted_error, controls,
              success, failed, cfg, holding=False, support_time=0.,
              step_duration=0., launch_quality=0., launch_motion=0.,
              release_progress=0.):
    terms = dict(time=-cfg.basket_catch_time_cost, parking=0., parking_progress=0.,
                 braking=0., aim=0., flight_progress=0., control=0.,
                 release=0., release_dir=0., forward=0., launch_quality=0.,
                 launch_motion=0., release_progress=0., hold=0.,
                 success=cfg.basket_w_score if success else 0.,
                 failure=-cfg.basket_catch_failure_cost if failed and not success else 0.)
    if phase == 'parking':
        terms['parking'] = -parking['distance']
        terms['parking_progress'] = cfg.basket_track_w_progress * (
            previous_parking_distance - parking['distance'])
        terms['braking'] = -float(np.sum((parking['velocity'] - parking['desired_velocity']) ** 2))
    elif phase == 'preparing':
        # Do not punish the landing point while the ball is still supported.
        terms['aim'] = 0. if holding else -cfg.basket_catch_w_aim * predicted_error
        terms['launch_quality'] = cfg.basket_w_launch_quality * launch_quality if holding else 0.
        # Action magnitude alone rewards flailing, not acceleration of the ball.
        terms['launch_motion'] = 0.
        terms['release_progress'] = (2.0 * float(np.clip(release_progress, 0., 1.))
                                     * launch_quality if holding else 0.)
        terms['control'] = -cfg.basket_throw_ctrl_arm * float(np.sum(np.asarray(controls['arm']) ** 2))
        terms['control'] -= cfg.basket_throw_ctrl_hand * float(np.sum(np.asarray(controls['hand']) ** 2))
        if holding and step_duration > 0.:
            terms['hold'] = -cfg.basket_hold_penalty_rate * min(
                step_duration, max(0., support_time + step_duration -
                                   cfg.basket_support_budget_seconds))
    elif phase == 'flight' and previous_flight_distance is not None:
        terms['flight_progress'] = cfg.basket_w_approach * (previous_flight_distance - distance)
    return float(sum(terms.values())), terms


def throw_quality(position, velocity, center, gravity, cfg):
    times = np.linspace(.25, 1.5, 64)
    velocities = (np.asarray(center) - position) / times[:, None] - .5 * times[:, None] * gravity
    normal = np.array([0., -np.sin(np.radians(cfg.basket_tilt_deg)), np.cos(np.radians(cfg.basket_tilt_deg))])
    descending = ((velocities + times[:, None] * gravity) @ normal) < 0.
    allowed = descending & (velocities[:, 2] >= cfg.basket_min_upward_release_speed) & (np.linalg.norm(velocities, axis=1) <= cfg.basket_reference_max_speed)
    if not np.any(allowed):
        return 0., np.zeros(3)
    candidates = velocities[allowed]
    errors = np.sum((candidates - velocity) ** 2, axis=1)
    index = int(np.argmin(errors))
    return float(np.exp(-errors[index] / cfg.basket_velocity_sigma ** 2)), candidates[index]


def underhand_geometry(palm, base, ball, center, velocity, cfg):
    reach = float(np.linalg.norm(np.asarray(palm)[:2] - np.asarray(base)[:2]))
    direction = np.asarray(center)[:2] - np.asarray(ball)[:2]
    forward = float(np.dot(np.asarray(velocity)[:2], direction / max(np.linalg.norm(direction), 1e-6)))
    upward = float(velocity[2])
    valid = (reach <= cfg.basket_arm_horizontal_reach and
             forward >= cfg.basket_min_forward_release_speed and
             upward >= cfg.basket_min_upward_release_speed)
    cost = -cfg.basket_arm_reach_cost * min(4., (max(0., reach-cfg.basket_arm_horizontal_reach) / .15)**2)
    return valid, cost, reach, forward, upward


def release_phase(phase, touching, had_contact, no_contact_seconds,
                  contact_seconds, clearance, cfg):
    """Pure physics-step transition; contact flicker is not a terminal release."""
    if touching:
        if phase == 'preparing' or contact_seconds >= cfg.basket_regrasp_seconds:
            return 'preparing'
        return 'recovering'
    if not had_contact:
        return 'preparing'
    if phase == 'flight':
        return 'flight'
    if (no_contact_seconds >= cfg.basket_release_debounce_seconds
            and clearance >= cfg.basket_release_clearance):
        return 'flight'
    return 'release_pending'


def ballistic_quality(position, velocity, center, gravity, cfg):
    """Correct-direction hoop-plane error; continuous near-miss fallback before floor.

    Exact quadratic plane roots avoid missing the opening between samples.
    Fallback quality is capped below a true crossing and never defines success.
    """
    p, v, c, g = map(lambda x: np.asarray(x, dtype=float),
                      (position, velocity, center, gravity))
    horizon = cfg.basket_prediction_horizon
    ground = cfg.basket_floor_z + cfg.basket_ball_radius
    def roots(a, b, d):
        if abs(a) < 1e-10:
            return [] if abs(b) < 1e-10 else [-d/b]
        disc = b*b - 4*a*d
        if disc < 0.:
            return []
        return [(-b-np.sqrt(disc))/(2*a), (-b+np.sqrt(disc))/(2*a)]
    floor_times = [t for t in roots(.5*g[2], v[2], p[2]-ground)
                   if t > 1e-8 and v[2]+g[2]*t < 0.]
    if floor_times:
        horizon = min(horizon, min(floor_times))
    normal = np.array([0., -np.sin(np.radians(cfg.basket_tilt_deg)),
                       np.cos(np.radians(cfg.basket_tilt_deg))])
    times = [t for t in roots(.5*np.dot(g, normal), np.dot(v, normal),
                              np.dot(p-c, normal))
             if 1e-8 < t <= horizon and np.dot(v+g*t, normal) < 0.]
    if times:
        error = min(float(np.linalg.norm(p+v*t+.5*g*t*t-c)) for t in times)
        quality = 1. / (1. + (error/cfg.basket_quality_sigma)**2)
        return float(quality), error, True
    times = np.linspace(0., max(0., horizon), 96)
    path = p + times[:, None]*v + .5*times[:, None]**2*g
    error = float(np.min(np.linalg.norm(path-c, axis=1)))
    quality = min(cfg.basket_near_miss_cap,
                  1. / (1. + (error/cfg.basket_quality_sigma)**2))
    return float(quality), error, False
