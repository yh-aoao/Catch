"""Physical throwing diagnostic. No policy, no ball position/velocity assistance."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np


def smooth_segment(t, duration):
    u = np.clip(t / duration, 0., 1.)
    return u*u*u*(10. + u*(-15. + 6.*u))


def trajectory(t, hold, back_time, swing_time, back, forward):
    if t <= hold:
        return np.zeros_like(back)
    if t <= hold + back_time:
        return smooth_segment(t-hold, back_time)*back
    return back + smooth_segment(t-hold-back_time, swing_time)*(forward-back)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--mode', choices=['hold', 'cartesian', 'joint'], default='cartesian')
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--hold', type=float, default=.5)
    parser.add_argument('--back-time', type=float, default=.5)
    parser.add_argument('--swing-time', type=float, default=.45)
    parser.add_argument('--release-fraction', type=float, default=.7)
    parser.add_argument('--open-time', type=float, default=.12)
    parser.add_argument('--back-y', type=float, default=-.08)
    parser.add_argument('--forward-y', type=float, default=.16)
    parser.add_argument('--up-z', type=float, default=.12)
    parser.add_argument('--joint-back', type=float, nargs=6, default=[0., -.15, .15, 0., 0., 0.])
    parser.add_argument('--joint-forward', type=float, nargs=6, default=[0., .25, -.25, 0., 0., 0.])
    parser.add_argument('--log-file', default='outputs/basket_probe.jsonl')
    args = parser.parse_args()
    if min(args.back_time, args.swing_time, args.open_time) <= 0 or args.hold < 0 or args.episodes < 1:
        parser.error('durations must be positive, hold nonnegative, episodes >= 1')
    if not 0 <= args.release_fraction <= 1:
        parser.error('release-fraction must be in [0,1]')
    os.chdir(Path(__file__).resolve().parent)
    # Import only after CLI parsing; --help does not require MuJoCo.
    import mujoco
    from gym_dcmm.envs.DcmmVecEnv import DcmmVecEnv
    env = DcmmVecEnv(task='Catching', object_motion='throw_basket',
        basket_fixed_base_training=True, basket_log=True, viewer=args.viewer,
        imshow_cam=False, render_per_step=False, object_eval=False)
    output = Path(args.log_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    hand_indices = np.array([0, 2, 3, 4, 6, 7, 8, 10, 11, 13, 14, 15])
    try:
        with output.open('w', encoding='utf-8') as log:
            log.write(json.dumps(dict(type='config', **vars(args)))+'\n')
            for episode in range(args.episodes):
                env.reset()
                start = float(env.Dcmm.data.time)
                q0 = env.Dcmm.data.qpos[15:21].copy()
                h0 = env.Dcmm.target_hand_qpos.copy()
                mujoco.mj_fwdPosition(env.Dcmm.model_arm, env.Dcmm.data_arm)
                p0 = env.Dcmm.data_arm.body('link6').xpos.copy()
                # Joint addresses are queried for clipping, not guessed from geom ids.
                joint_ids = [int(np.where(env.Dcmm.model.jnt_qposadr == i)[0][0]) for i in range(15,21)]
                if args.mode == 'joint':
                    back, forward = np.array(args.joint_back), np.array(args.joint_forward)
                else:
                    back, forward = np.array([0., args.back_y, -.02]), np.array([0., args.forward_y, args.up_z])
                release_time = args.hold + args.back_time + args.release_fraction*args.swing_time
                dt = env.steps_per_policy*env.Dcmm.model.opt.timestep
                while True:
                    t = float(env.Dcmm.data.time)-start
                    offset = trajectory(t+dt, args.hold, args.back_time, args.swing_time, back, forward)
                    arm = np.zeros(6)
                    if args.mode == 'joint':
                        target = q0 + offset
                        for j, jid in enumerate(joint_ids):
                            if env.Dcmm.model.jnt_limited[jid]:
                                target[j] = np.clip(target[j], *env.Dcmm.model.jnt_range[jid])
                        env._basket_joint_probe_target = target
                    elif args.mode == 'cartesian':
                        env.Dcmm.data_arm.qpos[:6] = env.Dcmm.data.qpos[15:21]
                        mujoco.mj_fwdPosition(env.Dcmm.model_arm, env.Dcmm.data_arm)
                        arm[:3] = np.clip(p0+offset-env.Dcmm.data_arm.body('link6').xpos, -.025, .025)
                    fraction = 0. if args.mode == 'hold' else smooth_segment(t+dt-release_time, args.open_time)
                    desired_hand = (1.-fraction)*h0
                    hand = np.clip((desired_hand-env.Dcmm.target_hand_qpos)[hand_indices], -.15, .15)
                    _, reward, terminated, truncated, info = env.step(dict(base=np.zeros(2), arm=arm, hand=hand))
                    record = dict(episode=episode, time=t+dt, mode=args.mode, reward=float(reward),
                        terminated=bool(terminated), truncated=bool(truncated),
                        reason=env.terminated_reason, phase=env.basket_phase,
                        arm_q=env.Dcmm.data.qpos[15:21].tolist(),
                        arm_target=env.Dcmm.target_arm_qpos.tolist(),
                        ball_position=env.Dcmm.data.qpos[37:40].tolist(),
                        control=info.get('basket_control', {}), release=info.get('basket_release'))
                    log.write(json.dumps(record)+'\n')
                    if terminated or truncated:
                        print(json.dumps(record, indent=2))
                        log.flush()
                        break
                    if args.viewer:
                        time.sleep(dt)
    finally:
        env.close()
    print('Saved:', output.resolve())


if __name__ == '__main__':
    main()
