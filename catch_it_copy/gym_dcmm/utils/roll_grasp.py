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
