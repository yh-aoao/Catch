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
        state['candidate'] = dict(position=p.copy(), time=float(now), direction=direction, velocity=np.asarray(velocity, dtype=float).copy(),
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


def reward(state, done, cfg, robot_collision=False, arm_action=None):
    terms = dict(valid_launch=state.pop('launch_pending', 0.),
                 flight_progress=state.pop('progress_pending', 0.), failure=0., robot_collision=0., arm_smoothness=0.)
    if arm_action is not None:
        current = np.asarray(arm_action, dtype=float).copy()
        previous = state.get('previous_arm_action')
        if previous is not None:
            terms['arm_smoothness'] = -cfg.basket_simple_smooth_weight * float(np.mean((current-previous)**2))
        state['previous_arm_action'] = current
    if robot_collision and not state.get('collision_paid'):
        terms['robot_collision'] = -cfg.basket_simple_collision_cost
        state['collision_paid'] = True
    if done and not state.get('terminal_paid'):
        if not state.get('confirmed'):
            terms['failure'] = -cfg.basket_simple_failure_cost
        state['terminal_paid'] = True
    return float(sum(terms.values())), terms


def arm_self_contacts(model, data, min_force):
    """Actual loaded contacts only; no distance probes, adjacent links or fingers."""
    import mujoco
    names = {'link1', 'link2', 'link3', 'link4', 'link5', 'link6'}
    result = []
    for index, contact in enumerate(data.contact):
        a, b = int(contact.geom1), int(contact.geom2)
        ba, bb = int(model.geom_bodyid[a]), int(model.geom_bodyid[b])
        na, nb = model.body(ba).name, model.body(bb).name
        if na not in names or nb not in names or ba == bb:
            continue
        if model.body_parentid[ba] == bb or model.body_parentid[bb] == ba:
            continue
        if contact.dist > 0:
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, force)
        if force[0] <= min_force:
            continue
        result.append(dict(pair=[na, nb], geoms=[a, b],
                           distance=float(contact.dist), normal_force=float(force[0])))
    return result
