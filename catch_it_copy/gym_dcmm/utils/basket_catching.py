"""World-coordinate basket events and phase-specific throwing rewards."""
import numpy as np


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
          step_duration=0., launch_quality=0., launch_motion=0.):
    terms = dict(time=-cfg.basket_catch_time_cost, parking=0., parking_progress=0.,
                 braking=0., aim=0., flight_progress=0., control=0.,
                 release=0., release_dir=0., forward=0., launch_quality=0.,
                 launch_motion=0., hold=0.,
                 success=cfg.basket_w_score if success else 0.,
                 failure=-cfg.basket_catch_failure_cost if failed and not success else 0.)
    if phase == 'parking':
        terms['parking'] = -parking['distance']
        terms['parking_progress'] = cfg.basket_track_w_progress * (
            previous_parking_distance - parking['distance'])
        terms['braking'] = -float(np.sum((parking['velocity'] - parking['desired_velocity']) ** 2))
    elif phase == 'preparing':
        terms['aim'] = -cfg.basket_catch_w_aim * predicted_error
        terms['launch_quality'] = cfg.basket_w_launch_quality * launch_quality
        terms['launch_motion'] = cfg.basket_w_launch_motion * launch_motion
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
