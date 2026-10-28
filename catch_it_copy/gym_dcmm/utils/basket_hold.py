"""Reset-only collision-free placement and bounded held-ball aiming progress."""
import numpy as np

from gym_dcmm.utils.basket_aim import crossing_error


def clear_spawn(model, data, ball_geom, hand_geoms, qpos_address=37):
    """Lift the reset sphere only; after reset the ball is entirely physical."""
    import mujoco
    original = data.qpos[qpos_address:qpos_address+3].copy()
    for lift in np.arange(0., .081, .001):
        data.qpos[qpos_address:qpos_address+3] = original + [0., 0., lift]
        mujoco.mj_forward(model, data)
        gap = min(mujoco.mj_geomDistance(model, data, int(ball_geom), int(geom), .1, None)
                  for geom in hand_geoms)
        if gap >= .001:
            return dict(lift=float(lift), minimum_gap=float(gap))
    data.qpos[qpos_address:qpos_address+3] = original
    mujoco.mj_forward(model, data)
    raise ValueError('Could not place Basket sphere clear of hand within 8cm; inspect the grasp geometry')


def preparation_progress(state, position, velocity, center, gravity, cfg):
    error = crossing_error(position, velocity, center, gravity, cfg.basket_tilt_deg)
    quality = 0. if error is None else float(np.exp(-(error/cfg.basket_aim_sigma)**2))
    previous = state.setdefault('best_held_quality', quality)
    state['held_aim_pending'] = state.get('held_aim_pending', 0.) + cfg.mobile_hold_aim_bonus*max(0., quality-previous)
    state['best_held_quality'] = max(previous, quality)


def arm_drive_progress(state, palm_relative, proximal_velocity, ball_velocity, direction, bonus):
    """Bounded forward carrying credit; wrist rotation/base motion alone cannot earn it.

    Caller supplies shoulder/elbow-only Jacobian velocity and calls only during
    supported, unlocked, collision-free pre-release motion. Targets are scales,
    not prescribed trajectories. Release credit uses LAST contact, not best ever.
    """
    origin = state.setdefault('arm_drive_origin', np.asarray(palm_relative).copy())
    forward = float(np.dot(np.asarray(palm_relative)-origin, direction))
    arm_speed = float(np.dot(proximal_velocity, direction))
    ball_speed = float(np.dot(ball_velocity, direction))
    quality = float(np.clip(forward/.12, 0., 1.) * np.clip(arm_speed/.6, 0., 1.)
                    * np.clip(ball_speed/.6, 0., 1.))
    previous = state.get('best_arm_drive', 0.)
    state['arm_drive_pending'] = state.get('arm_drive_pending', 0.) + bonus*max(0., quality-previous)
    state['best_arm_drive'] = max(previous, quality)
    state['last_contact_arm_drive'] = quality
    state['arm_drive_forward'] = forward
    state['arm_drive_speed'] = arm_speed


def measure_arm_drive(model, data, state, center, bonus):
    import mujoco
    palm, base = data.body('link6'), data.body('arm_base')
    direction = np.asarray(center)-base.xpos
    direction[2] = 0.
    direction /= max(float(np.linalg.norm(direction)), 1e-9)
    jac = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jac, None, palm.id)
    columns = [int(model.jnt_dofadr[np.flatnonzero(model.jnt_qposadr == q)[0]]) for q in (15,16,17)]
    arm_drive_progress(state, palm.xpos-base.xpos, jac[:,columns] @ data.qvel[columns],
                       data.qvel[36:39], direction, bonus)


def arm_drive_reward(state, launch, bonus, robot_collision=False):
    pending = state.pop('arm_drive_pending', 0.)
    terms = dict(arm_drive_progress=0. if robot_collision else pending, arm_drive_release=0.)
    if launch.get('confirmed') and not robot_collision and not state.get('release_paid'):
        terms['arm_drive_release'] = bonus*state.get('last_contact_arm_drive',0.)
        state['release_paid'] = True
    return terms
