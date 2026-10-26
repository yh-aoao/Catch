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


def direct_launch_progress(state, velocity, reference, position, center):
    """Separate bounded exploration credit from coordinated launch credit."""
    velocity, reference, position, center = map(lambda x: np.asarray(x, float),
                                               (velocity, reference, position, center))
    direction = center[:2] - position[:2]
    direction /= max(float(np.linalg.norm(direction)), 1e-8)
    forward = float(velocity[:2] @ direction)
    f = np.clip(forward / max(.5, float(reference[:2] @ direction)), 0., 1.)
    u = np.clip(velocity[2] / max(.5, float(reference[2])), 0., 1.)
    lateral = np.linalg.norm(velocity[:2] - forward * direction)
    aligned = float(np.exp(-lateral**2))
    return sum(weight * best_progress(state, key, value * aligned)
               for weight, key, value in
               ((.25, 'v5_forward', f), (.25, 'v5_up', u),
                (.5, 'v5_coordinated', np.sqrt(f*u))))


def retention_credit(state, local_position, dt, slip_scale=.25, budget=1.):
    """Contact-gated, finite support credit; not an incentive to hold forever."""
    local_position = np.asarray(local_position, float)
    previous = state.get('v5_local_position')
    state['v5_local_position'] = local_position.copy()
    if previous is None:
        return 0., 0.
    slip = float(np.linalg.norm(local_position - previous) / dt)
    duration = min(dt, max(0., budget - state.get('v5_support_seconds', 0.)))
    state['v5_support_seconds'] = state.get('v5_support_seconds', 0.) + duration
    return duration * float(np.exp(-(slip/slip_scale)**2)), slip


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
