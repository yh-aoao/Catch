"""Bounded finger shaping: leave an entrance before rewarding enclosure."""
import numpy as np


def grasp_terms(joints, local_ball, contact, cfg):
    # link6 local Y is the palm normal; X/Z span the palm.
    local_ball = np.asarray(local_ball)
    inside = (np.linalg.norm(local_ball[[0, 2]]) <= cfg.roll_grasp_entry_radius
              and -0.02 <= local_ball[1] <= cfg.roll_grasp_entry_depth)
    ready = bool(inside)
    fingers = np.asarray(joints)[[0, 2, 3, 4, 6, 7, 8, 10, 11, 13, 14, 15]].reshape(4, 3)
    # Continuous transition avoids repeated open/close targets at a sharp boundary.
    radial = np.linalg.norm(local_ball[[0, 2]])
    gate = np.clip((cfg.roll_grasp_entry_radius + .03 - radial) / .03, 0., 1.)
    gate *= np.clip((cfg.roll_grasp_entry_depth + .04 - local_ball[1]) / .04, 0., 1.)
    gate *= np.clip((local_ball[1] + .05) / .03, 0., 1.)
    target = cfg.roll_grasp_open_target + gate * (cfg.roll_grasp_closed_target-cfg.roll_grasp_open_target)
    posture = float(np.exp(-np.mean((fingers - target) ** 2) / 0.25))
    # Bounded target rather than rewarding unlimited flexion.
    terms = {'posture': cfg.roll_grasp_posture_weight * posture,
             'coordination': -cfg.roll_grasp_chain_weight * float(
                 np.mean(np.minimum(np.diff(fingers, axis=1) ** 2, 1.))) if ready else 0.,
             'enclosure': float(contact and ready) * posture}
    return ready, terms


def hand_stability_terms(velocity, previous_velocity, dt, settled, cfg):
    velocity = np.asarray(velocity)
    # Allow decisive closing; penalize high speed mainly once the ball settles.
    weight = cfg.roll_hand_speed_weight * (1. if settled else .2)
    speed = -weight * float(np.mean(np.minimum((velocity / 3.)**2, 4.)))
    acceleration = 0.
    if previous_velocity is not None:
        delta = (velocity - previous_velocity) / max(dt, 1e-6)
        acceleration = -cfg.roll_hand_accel_weight * float(np.mean(np.minimum((delta / 50.)**2, 4.)))
    return dict(hand_speed=speed, hand_acceleration=acceleration)


def smooth_target_delta(delta, previous, dt, settled, cfg, phase=None):
    """Filter policy joint-target increments, preserving faster capture motion."""
    delta = np.asarray(delta, dtype=float)
    previous = np.zeros_like(delta) if previous is None else np.asarray(previous)
    rate = cfg.roll_target_hold_rate if settled else cfg.roll_target_capture_rate
    if phase == "capturing":
        rate = cfg.roll_grip_capture_rate
    limit = rate * dt
    desired = np.clip(delta, -limit, limit)
    alpha = cfg.roll_target_hold_alpha if settled else cfg.roll_target_capture_alpha
    if phase == "capturing":
        alpha = cfg.roll_grip_capture_alpha
    applied = np.clip(alpha*desired + (1-alpha)*previous, -limit, limit)
    scale = max(limit, 1e-6)
    terms = dict(target_motion=-cfg.roll_target_motion_weight * float(np.mean((applied/scale)**2)),
                 target_change=-cfg.roll_target_change_weight * float(np.mean(np.minimum(((desired-previous)/scale)**2, 4.))))
    if phase == "capturing":
        terms = {key: value * cfg.roll_grip_capture_regularization for key, value in terms.items()}
    return applied, terms


def grip_phase(state, contact, clear, speed, now, cfg):
    if contact and clear:
        if state.get('first_contact') is None:
            state['first_contact'] = now
        state['last_contact'] = now
    last = state.get('last_contact', -float('inf'))
    active = (clear and state.get('first_contact') is not None
              and now-last <= cfg.roll_grip_contact_grace)
    if not active:
        state['first_contact'] = None
        # Invalidate the grace window too: clear can recover before a new contact.
        state.pop('last_contact', None)
        return 'waiting'
    age = now-state['first_contact']
    return ('holding' if age >= cfg.roll_grip_capture_seconds and speed <= .25
            else 'capturing')


def finger_contact_count(model, contacts, palm_geom, hand_ids):
    """Count contacted finger kinematic branches, not collision mesh count."""
    root = int(model.geom_bodyid[palm_geom])
    branches = set()
    for geom in set(map(int, contacts)).intersection(map(int, hand_ids)):
        body = int(model.geom_bodyid[geom])
        if body == root:
            continue
        while body and int(model.body_parentid[body]) != root:
            body = int(model.body_parentid[body])
        if body and int(model.body_parentid[body]) == root:
            branches.add(body)
    return len(branches)


def grip_feedback(state, contact, clear, distance, speed, fingers, dt, cfg):
    retained = bool(contact and clear and distance <= cfg.roll_drop_distance)
    previous = state.get('duration', 0.)
    state['duration'] = previous+dt if retained else 0.
    if retained:
        state['lost'] = 0.
        state['had_retention'] = True
    elif state.get('had_retention', False):
        state['lost'] = state.get('lost', 0.)+dt
    credit = min(state['duration'], cfg.roll_grip_retention_seconds)
    best = state.get('best_duration', 0.)
    progress = cfg.roll_grip_retention_weight*max(0., credit-best)
    state['best_duration'] = max(best, credit)
    dropped = (state.get('had_retention', False) and
               state.get('lost', 0.) >= cfg.roll_grip_contact_grace and
               not state.get('drop_penalized', False))
    if dropped:
        state['drop_penalized'] = True
    # Continuous contact quality: no exact closed-angle requirement.
    enclosure = (cfg.roll_grip_enclosure_weight*min(fingers/2., 1.)*
                 np.exp(-(speed/.35)**2) if retained else 0.)
    return dict(finger_contact=enclosure, retention_progress=progress,
                slip=-cfg.roll_grip_drop_cost if dropped else 0.)
