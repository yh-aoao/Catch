"""Velocity-action conversion and bounded acceleration credit; no ball actuation."""
import numpy as np


def joint_velocity_target(measured, scaled_action, action_scale, speed_limits, dt, limits):
    q, action, speed, bounds = map(lambda x: np.asarray(x, float),
                                   (measured, scaled_action, speed_limits, limits))
    if q.shape != (6,) or action.shape != (6,) or speed.shape != (6,) or bounds.shape != (6, 2):
        raise ValueError('Expected six arm joints and their limits')
    if action_scale <= 0 or dt <= 0 or not all(np.isfinite(x).all() for x in (q, action, speed, bounds)) or np.any(speed <= 0):
        raise ValueError('Invalid velocity control inputs')
    normalized = np.clip(action/action_scale, -1., 1.)
    return np.clip(q + normalized*speed*dt, bounds[:, 0], bounds[:, 1])


def acceleration_readiness(velocity, reference, position, center):
    velocity, reference, position, center = map(lambda x: np.asarray(x, float), (velocity, reference, position, center))
    direction = center[:2]-position[:2]
    direction /= max(float(np.linalg.norm(direction)), 1e-8)
    forward = float(velocity[:2] @ direction)
    needed_forward = max(.5, float(reference[:2] @ direction))
    needed_up = max(.5, float(reference[2]))
    # Early forward or upward acceleration is learnable; full credit needs both.
    return .5*float(np.clip(forward/needed_forward, 0., 1.)) + .5*float(np.clip(velocity[2]/needed_up, 0., 1.))


def best_progress(state, key, value):
    """Nonnegative, episode-bounded credit. Recontact cannot reset the budget."""
    value = float(np.clip(value, 0., 1.))
    previous = state.get(key, 0.)
    state[key] = max(previous, value)
    return max(0., value-previous)


def coupled_readiness(velocity, reference, position, center):
    velocity, reference, position, center = map(lambda x: np.asarray(x, float), (velocity, reference, position, center))
    direction = center[:2] - position[:2]
    direction /= max(float(np.linalg.norm(direction)), 1e-8)
    forward = float(velocity[:2] @ direction)
    lateral = float(np.linalg.norm(velocity[:2] - forward * direction))
    f = np.clip(forward / max(.5, float(reference[:2] @ direction)), 0., 1.)
    u = np.clip(velocity[2] / max(.5, float(reference[2])), 0., 1.)
    return float(np.sqrt(f*u) * np.exp(-(lateral/1.)**2))


def confirm_throw(candidate, position, now, collision, min_time=.1, min_travel=.15):
    if candidate is None:
        return False, dict(reason='no_candidate')
    if collision:
        candidate['invalidated'] = True
    elapsed = float(now-candidate['time'])
    travel = float((np.asarray(position)[:2]-np.asarray(candidate['position'])[:2]) @ np.asarray(candidate['direction']))
    reason = ('collision' if candidate.get('invalidated') else
              'release_velocity' if not candidate['eligible'] else
              'flight_duration' if elapsed < min_time else
              'flight_travel' if travel < min_travel else 'confirmed')
    return reason == 'confirmed', dict(reason=reason, flight_seconds=elapsed, forward_travel=travel,
                                      forward_speed=candidate['forward'], upward_speed=candidate['upward'])
