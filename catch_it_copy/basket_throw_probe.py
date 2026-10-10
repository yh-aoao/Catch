"""Physical throwing diagnostic. No policy, no ball position/velocity assistance."""
import argparse
import json
import os
from pathlib import Path
import time
from itertools import product
import numpy as np


def smooth_segment(t, duration):
    u = np.clip(t / duration, 0., 1.)
    return u*u*u*(10. + u*(-15. + 6.*u))


def trajectory(t, hold, back_time, swing_time, back, forward):
    if t <= hold:
        return np.zeros_like(back)
    if back_time <= 0:
        return smooth_segment(t-hold, swing_time)*forward
    if t <= hold + back_time:
        return smooth_segment(t-hold, back_time)*back
    return back + smooth_segment(t-hold-back_time, swing_time)*(forward-back)


def trial_settings(args):
    """Each scan cell gets the same number of independent reset trials."""
    pairs = product(args.scan_swing_times, args.scan_release_fractions) if args.scan else [(args.swing_time, args.release_fraction)]
    return [(float(swing), float(release)) for swing, release in pairs
            for _ in range(args.episodes)]


def release_diagnostic(snapshot, start, swing_start, planned_open, direction):
    """Descriptive physical diagnostic, separate from environment success."""
    if not snapshot:
        return dict(status='no_free_flight_detected')
    last = snapshot['last_contact']
    velocity = np.asarray(last['velocity'])
    actual_release = float(last['time'])-start
    forward_speed = float(np.dot(velocity[:2], direction))
    upward_speed = float(velocity[2])
    status = ('lost_before_forward_swing' if actual_release < swing_start else
              'lost_before_planned_open' if actual_release < planned_open else 'released_after_planned_open')
    return dict(status=status, last_contact_time=actual_release,
                opening_time_error=actual_release-planned_open,
                forward_speed=forward_speed, upward_speed=upward_speed,
                upward_forward_flight=bool(forward_speed > .1 and upward_speed > .1),
                note='upward_forward_flight is a diagnostic, not basket success')


def joint_direction_report(robot, mujoco, joint_ids, amplitude):
    """Static FK directions only; does not actuate the robot or move the ball."""
    saved = robot.data_arm.qpos.copy()
    q0 = robot.data.qpos[15:21].copy()
    records = []
    try:
        robot.data_arm.qpos[:6] = q0
        mujoco.mj_fwdPosition(robot.model_arm, robot.data_arm)
        p0 = robot.data_arm.body('link6').xpos.copy()
        for j, jid in enumerate(joint_ids):
            for sign in [-1., 1.]:
                target = q0.copy()
                target[j] += sign*amplitude
                if robot.model.jnt_limited[jid]:
                    target[j] = np.clip(target[j], *robot.model.jnt_range[jid])
                robot.data_arm.qpos[:6] = target
                mujoco.mj_fwdPosition(robot.model_arm, robot.data_arm)
                body = robot.data_arm.body('link6')
                records.append(dict(joint=j+1, delta=float(target[j]-q0[j]),
                    ee_delta_arm_frame=(body.xpos-p0).tolist(),
                    link6_y_arm_frame=body.xmat.reshape(3,3)[:,1].tolist()))
    finally:
        robot.data_arm.qpos[:] = saved
        mujoco.mj_fwdPosition(robot.model_arm, robot.data_arm)
    return records


def solve_absolute_probe_pose(robot, mujoco, position, quaternion, previous_target, joint_ids):
    """Measured IK seed, but fixed absolute pose goal; retain target on failure."""
    measured = robot.data.qpos[15:21].copy()
    try:
        robot.data_arm.qpos[:6] = measured
        mujoco.mj_fwdPosition(robot.model_arm, robot.data_arm)
        solution, success = robot.ik_arm_solve(position.copy(), quaternion.copy())
        solution = np.asarray(solution, dtype=float)
        valid = bool(success) and solution.shape == (6,) and bool(np.all(np.isfinite(solution)))
        if valid:
            valid = all(not robot.model.jnt_limited[jid] or
                robot.model.jnt_range[jid,0] <= solution[j] <= robot.model.jnt_range[jid,1]
                for j, jid in enumerate(joint_ids))
        return (solution.copy() if valid else previous_target.copy()), valid
    finally:
        robot.data_arm.qpos[:6] = measured
        mujoco.mj_fwdPosition(robot.model_arm, robot.data_arm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--mode', choices=['hold', 'cartesian', 'joint'], default='cartesian')
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--hold', type=float, default=.5)
    parser.add_argument('--back-time', type=float, default=0., help='0 skips back-swing entirely')
    parser.add_argument('--swing-time', type=float, default=.45)
    parser.add_argument('--release-fraction', type=float, default=.5)
    parser.add_argument('--open-time', type=float, default=.12)
    parser.add_argument('--no-release', action='store_true', help='swing with initial hand target; no scripted opening')
    parser.add_argument('--back-y', type=float, default=-.08)
    parser.add_argument('--forward-y', type=float, default=.16)
    parser.add_argument('--up-z', type=float, default=.12)
    parser.add_argument('--joint-back', type=float, nargs=6, default=[0., -.15, .15, 0., 0., 0.])
    parser.add_argument('--joint-forward', type=float, nargs=6, default=[0., .25, -.25, 0., 0., 0.])
    parser.add_argument('--log-file', default='outputs/basket_probe.jsonl')
    parser.add_argument('--scan', action='store_true', help='scan swing/opening timing; episodes per cell')
    parser.add_argument('--scan-swing-times', type=float, nargs='+', default=[.3, .45, .6])
    parser.add_argument('--scan-release-fractions', type=float, nargs='+', default=[.3, .5, .7])
    parser.add_argument('--calibrate-joints', action='store_true', help='print static FK directions and exit')
    parser.add_argument('--calibration-step', type=float, default=.1)
    args = parser.parse_args()
    if min(args.swing_time, args.open_time, args.calibration_step) <= 0 or min(args.hold, args.back_time) < 0 or args.episodes < 1:
        parser.error('swing/open/calibration step must be positive; hold/back nonnegative; episodes >= 1')
    if not 0 <= args.release_fraction <= 1:
        parser.error('release-fraction must be in [0,1]')
    if any(t <= 0 for t in args.scan_swing_times) or any(not 0 <= f <= 1 for f in args.scan_release_fractions):
        parser.error('scan swing times must be positive and release fractions in [0,1]')
    if args.scan and args.mode == 'hold':
        parser.error('hold mode has no swing/opening to scan')
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
            for episode, (swing_time, release_fraction) in enumerate(trial_settings(args)):
                env.reset()
                start = float(env.Dcmm.data.time)
                q0 = env.Dcmm.data.qpos[15:21].copy()
                h0 = env.Dcmm.target_hand_qpos.copy()
                env.Dcmm.data_arm.qpos[:6] = q0
                mujoco.mj_fwdPosition(env.Dcmm.model_arm, env.Dcmm.data_arm)
                p0 = env.Dcmm.data_arm.body('link6').xpos.copy()
                quat0 = env.Dcmm.data_arm.body('link6').xquat.copy()
                pose_command = p0.copy()
                joint_command = q0.copy()
                probe_ik_attempts = probe_ik_successes = 0
                # Joint addresses are queried for clipping, not guessed from geom ids.
                joint_ids = [int(np.where(env.Dcmm.model.jnt_qposadr == i)[0][0]) for i in range(15,21)]
                if args.calibrate_joints:
                    report = dict(type='joint_directions', frame='arm_model',
                        warning='static FK only; verify dynamic motion and collisions with probe',
                        candidates=joint_direction_report(env.Dcmm, mujoco, joint_ids, args.calibration_step))
                    log.write(json.dumps(report)+'\n')
                    print(json.dumps(report, indent=2))
                    break
                if args.mode == 'joint':
                    back, forward = np.array(args.joint_back), np.array(args.joint_forward)
                else:
                    back, forward = np.array([0., args.back_y, -.02]), np.array([0., args.forward_y, args.up_z])
                release_time = args.hold + args.back_time + release_fraction*swing_time
                dt = env.steps_per_policy*env.Dcmm.model.opt.timestep
                initial_ball = env.Dcmm.data.qpos[37:40].copy()
                direction = env.basket_center[:2]-initial_ball[:2]
                direction /= max(float(np.linalg.norm(direction)), 1.e-9)
                previous_ee = env.Dcmm.data.body('link6').xpos.copy()
                initial_ee = previous_ee.copy()
                previous_time = start
                max_ee_forward = max_ee_up = max_ball_forward = max_ball_up = 0.
                log.write(json.dumps(dict(type='trial', episode=episode, swing_time=swing_time,
                    release_fraction=release_fraction, planned_open_time=release_time,
                    forward_swing_time=args.hold+args.back_time))+'\n')
                print(f'[probe-trial] episode={episode} swing={swing_time:.3f} open_fraction={release_fraction:.2f}', flush=True)
                while True:
                    t = float(env.Dcmm.data.time)-start
                    offset = trajectory(t+dt, args.hold, args.back_time, swing_time, back, forward)
                    arm = np.zeros(6)
                    probe_ik_ok = None
                    if args.mode == 'hold' or t+dt <= args.hold:
                        # Zero Cartesian increments would re-anchor the joint target
                        # to each new measured pose. Keep the ORIGINAL target instead.
                        joint_command = q0.copy()
                        pose_command = p0.copy()
                    elif args.mode == 'joint':
                        target = q0 + offset
                        for j, jid in enumerate(joint_ids):
                            if env.Dcmm.model.jnt_limited[jid]:
                                target[j] = np.clip(target[j], *env.Dcmm.model.jnt_range[jid])
                        joint_command = target
                    elif args.mode == 'cartesian':
                        # Rate-limit the commanded path, not the error from a sagging
                        # measured pose. Hold the initial orientation throughout.
                        candidate_pose = pose_command + np.clip(p0+offset-pose_command, -.025, .025)
                        joint_command, probe_ik_ok = solve_absolute_probe_pose(env.Dcmm,
                            mujoco, candidate_pose, quat0, joint_command, joint_ids)
                        probe_ik_attempts += 1
                        probe_ik_successes += int(probe_ik_ok)
                        if probe_ik_ok:
                            pose_command = candidate_pose
                    # Existing diagnostic hook still uses the physical PID/contacts;
                    # PPO never sets this attribute. All probe modes now use it.
                    env._basket_joint_probe_target = joint_command.copy()
                    fraction = 0. if args.mode == 'hold' or args.no_release else smooth_segment(t+dt-release_time, args.open_time)
                    desired_hand = (1.-fraction)*h0
                    hand = np.clip((desired_hand-env.Dcmm.target_hand_qpos)[hand_indices], -.15, .15)
                    _, reward, terminated, truncated, info = env.step(dict(base=np.zeros(2), arm=arm, hand=hand))
                    # Refresh kinematics after integration; measure realized world motion.
                    mujoco.mj_fwdPosition(env.Dcmm.model, env.Dcmm.data)
                    actual_time = float(env.Dcmm.data.time)
                    body = env.Dcmm.data.body('link6')
                    ee = body.xpos.copy()
                    ee_velocity = (ee-previous_ee)/max(actual_time-previous_time, 1.e-9)
                    normal = body.xmat.reshape(3,3)[:,1].copy()
                    ball_velocity = env.Dcmm.data.qvel[36:39].copy()
                    env.Dcmm.data_arm.qpos[:6] = env.Dcmm.data.qpos[15:21]
                    mujoco.mj_fwdPosition(env.Dcmm.model_arm, env.Dcmm.data_arm)
                    ee_arm = env.Dcmm.data_arm.body('link6').xpos.copy()
                    actual_quat = env.Dcmm.data_arm.body('link6').xquat.copy()
                    orientation_error = 2.*np.arccos(np.clip(abs(float(np.dot(actual_quat, quat0))), 0., 1.))
                    max_ee_forward = max(max_ee_forward, float(np.dot(ee_velocity[:2], direction)))
                    max_ee_up = max(max_ee_up, float(ee_velocity[2]))
                    max_ball_forward = max(max_ball_forward, float(np.dot(ball_velocity[:2], direction)))
                    max_ball_up = max(max_ball_up, float(ball_velocity[2]))
                    previous_ee, previous_time = ee, actual_time
                    record = dict(type='step', episode=episode, time=actual_time-start, mode=args.mode, reward=float(reward),
                        terminated=bool(terminated), truncated=bool(truncated),
                        reason=env.terminated_reason, phase=env.basket_phase,
                        arm_q=env.Dcmm.data.qpos[15:21].tolist(),
                        arm_target=env.Dcmm.target_arm_qpos.tolist(),
                        probe_control_version='absolute_pose_v2',
                        ee_goal_arm_frame=(p0+offset if args.mode == 'cartesian' else p0).tolist() if args.mode != 'joint' else None,
                        ee_command_arm_frame=pose_command.tolist() if args.mode != 'joint' else None,
                        orientation_goal_wxyz=quat0.tolist() if args.mode != 'joint' else None,
                        probe_ik_ok=probe_ik_ok, probe_ik_attempts=probe_ik_attempts,
                        probe_ik_successes=probe_ik_successes,
                        ee_actual_arm_frame=ee_arm.tolist(),
                        ee_command_error=float(np.linalg.norm(ee_arm-pose_command)) if args.mode != 'joint' else None,
                        orientation_error_rad=float(orientation_error),
                        ee_position_world=ee.tolist(), ee_velocity_world=ee_velocity.tolist(),
                        ee_displacement_world=(ee-initial_ee).tolist(),
                        link6_y_world=normal.tolist(), palm_up_cos=float(normal[2]),
                        planned_open_fraction=float(fraction),
                        hand_q=env.Dcmm.data.qpos[21:37].tolist(), hand_target=env.Dcmm.target_hand_qpos.tolist(),
                        ball_position=env.Dcmm.data.qpos[37:40].tolist(),
                        ball_velocity=ball_velocity.tolist(),
                        control=info.get('basket_control', {}), release=info.get('basket_release'))
                    log.write(json.dumps(record)+'\n')
                    if terminated or truncated:
                        summary = dict(type='summary', episode=episode, swing_time=swing_time,
                            release_fraction=release_fraction, reason=env.terminated_reason,
                            success=bool(info.get('success', False)),
                            no_release=args.no_release, probe_control_version='absolute_pose_v2',
                            probe_ik_attempts=probe_ik_attempts, probe_ik_successes=probe_ik_successes,
                            max_ee_forward_speed=max_ee_forward, max_ee_upward_speed=max_ee_up,
                            max_ball_forward_speed=max_ball_forward, max_ball_upward_speed=max_ball_up,
                            release=release_diagnostic(info.get('basket_release'), start,
                                args.hold+args.back_time, release_time, direction))
                        log.write(json.dumps(summary)+'\n')
                        print('[probe-summary] '+json.dumps(summary), flush=True)
                        log.flush()
                        break
                    if args.viewer:
                        time.sleep(dt)
    finally:
        env.close()
    print('Saved:', output.resolve())


if __name__ == '__main__':
    main()
