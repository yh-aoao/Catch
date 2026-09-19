"""Bounded finger shaping: leave an entrance before rewarding enclosure."""
import numpy as np


def grasp_terms(joints, local_ball, contact, cfg):
    # link6 local Y is the palm normal; X/Z span the palm.
    local_ball = np.asarray(local_ball)
    inside = (np.linalg.norm(local_ball[[0, 2]]) <= cfg.roll_grasp_entry_radius
              and -0.02 <= local_ball[1] <= cfg.roll_grasp_entry_depth)
    ready = bool(inside)
    fingers = np.asarray(joints)[[0, 2, 3, 4, 6, 7, 8, 10, 11, 13, 14, 15]].reshape(4, 3)
    target = cfg.roll_grasp_closed_target if ready else cfg.roll_grasp_open_target
    posture = float(np.exp(-np.mean((fingers - target) ** 2) / 0.25))
    # Bounded target rather than rewarding unlimited flexion.
    terms = {'posture': cfg.roll_grasp_posture_weight * posture,
             'coordination': -cfg.roll_grasp_chain_weight * float(
                 np.mean(np.minimum(np.diff(fingers, axis=1) ** 2, 1.))) if ready else 0.,
             'enclosure': float(contact and ready) * posture}
    return ready, terms
