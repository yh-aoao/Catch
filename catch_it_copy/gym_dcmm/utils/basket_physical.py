"""Physical reset preparation and last-contact forward-throw qualification."""
import numpy as np


def release_quality(drive):
    # Last-contact values, never the largest displacement earlier in the swing.
    return float(min(np.clip(drive.get('arm_drive_forward', 0.)/.06, 0., 1.),
                     np.clip(drive.get('arm_drive_speed', 0.)/.30, 0., 1.)))


def settle(env):
    """Hold robot targets while gravity/contact settle the unactuated ball.

    This happens before the policy episode. No attachment, ball velocity reset,
    force assist, or pose correction is applied during the settling rollout.
    """
    import mujoco
    from gym_dcmm.utils.basket_grip import palm_coordinates
    from gym_dcmm.utils.basket_simple import arm_self_contacts
    from gym_dcmm.utils.roll_rewards import hand_collision_ids
    model, data = env.Dcmm.model, env.Dcmm.data
    hands = hand_collision_ids(model, env.hand_start_id)
    for name in ('base', 'arm', 'hand'):
        env.action_buffer[name].clear()
    env.update_target_ctrl()
    start = float(data.time)
    dt = float(model.opt.timestep)
    previous = palm_coordinates(data, data.qpos[37:40])
    stable = 0.
    while data.time-start < 1.5:
        data.ctrl[:-1] = env._get_ctrl()
        data.ctrl[-1] = 0.
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        contacts = env._get_contacts()
        object_contacts = contacts['object_contacts'].astype(int)
        touching = bool(np.any(np.isin(object_contacts, hands)))
        local = palm_coordinates(data, data.qpos[37:40])
        slip_speed = float(np.linalg.norm(local-previous)/dt)
        speed = float(np.linalg.norm(data.qvel[36:39]))
        previous = local
        if (np.any(~np.isin(object_contacts, hands)) or
                arm_self_contacts(model, data, 1.)):
            raise ValueError('Physical Basket preparation collided; ball was not repositioned')
        ready = touching and speed < .08 and slip_speed < .03 and 0.02 < local[1] < .075
        stable = stable+dt if ready else 0.
        if data.time-start >= .4 and stable >= .10:
            env.contacts = contacts
            env.object_pos3d = data.qpos[37:40].copy()
            return dict(seconds=float(data.time-start), stable_seconds=stable,
                        ball_speed=speed, slip_speed=slip_speed, local=local.tolist())
    raise ValueError('Physical Basket preparation did not reach stable palm support within 1.5s')
