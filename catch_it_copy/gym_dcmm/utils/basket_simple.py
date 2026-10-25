"""Launch-only reward, updated at physics frequency; no object actuation."""
import numpy as np


def update_launch(state, position, velocity, center, now, touching, collision, cfg):
    """One rewarded flight per episode. Any collision permanently ends its credit."""
    if state.get('stopped'):
        return
    p = np.asarray(position, dtype=float)
    if touching:
        if state.get('confirmed'):
            state['stopped'] = True
            return
        direction = np.asarray(center, dtype=float)[:2] - p[:2]
        direction /= max(float(np.linalg.norm(direction)), 1e-8)
        state['candidate'] = dict(position=p.copy(), time=float(now), direction=direction,
                                  forward=float(np.asarray(velocity)[:2] @ direction), invalid=False)
        # Touching the ground or another body while held is not free flight.
        state['candidate']['invalid'] = bool(collision)
        return
    candidate = state.get('candidate')
    if candidate is None:
        return
    if collision:
        candidate['invalid'] = True
        if state.get('confirmed'):
            state['stopped'] = True
        return
    if candidate['invalid']:
        return
    if not state.get('confirmed'):
        if (now-candidate['time'] < cfg.basket_simple_free_seconds or
                candidate['forward'] < cfg.basket_simple_forward_speed):
            return
        state['confirmed'] = True
        state['launch_pending'] = cfg.basket_simple_launch_bonus
    distance = max(0., float((p[:2]-candidate['position'][:2]) @ candidate['direction']))
    old = state.get('best_distance', 0.)
    best = max(old, distance)
    state['progress_pending'] = state.get('progress_pending', 0.) + cfg.basket_simple_progress_bonus * (
        min(best/cfg.basket_simple_target_distance, 1.) - min(old/cfg.basket_simple_target_distance, 1.))
    state['best_distance'] = best


def reward(state, done, cfg):
    terms = dict(valid_launch=state.pop('launch_pending', 0.),
                 flight_progress=state.pop('progress_pending', 0.), failure=0.)
    if done and not state.get('terminal_paid'):
        if not state.get('confirmed'):
            terms['failure'] = -cfg.basket_simple_failure_cost
        state['terminal_paid'] = True
    return float(sum(terms.values())), terms
