"""World-coordinate basket events and phase-specific throwing rewards."""
import numpy as np


def joint_throw_reward(task, phase, velocity, reference, holding, release_quality,
                       distance, previous_distance, controls, success, failed,
                       dt, state, cfg):
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
        norm = float(np.linalg.norm(reference))
        readiness = float(np.clip(1. - np.linalg.norm(velocity-reference) / norm, 0., 1.)) if norm > 1e-6 else 0.
        # Only a new best earns credit: oscillations/recontacts cannot farm it.
        best = state.get('best_readiness', 0.)
        terms['velocity_progress'] = cfg.basket_w_velocity_progress * max(0., readiness-best)
        state['best_readiness'] = max(best, readiness)
    if release_quality > 0. and not state.get('released', False):
        if not (failed and not success):
            terms['valid_release'] = cfg.basket_w_valid_release * float(np.clip(release_quality, 0., 1.))
        state['released'] = True
    if phase == 'flight' and previous_distance is not None:
        terms['flight_progress'] = cfg.basket_w_approach * (previous_distance-distance)
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
    allowed = descending & (np.linalg.norm(velocities, axis=1) <= cfg.basket_reference_max_speed)
    if not np.any(allowed):
        return 0., np.zeros(3)
    candidates = velocities[allowed]
    errors = np.sum((candidates - velocity) ** 2, axis=1)
    index = int(np.argmin(errors))
    return float(np.exp(-errors[index] / cfg.basket_velocity_sigma ** 2)), candidates[index]
