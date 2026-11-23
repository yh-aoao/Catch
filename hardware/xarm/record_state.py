#!/usr/bin/env python3
"""Read-only xArm recorder. Does not enable, clear errors, change modes, or move."""
import argparse
import datetime
import json
import math
from pathlib import Path
import time


def checked(call, label):
    code, data = call()
    if code != 0:
        raise RuntimeError('{} failed: code={}'.format(label, code))
    return data


def sample(arm, clock=time.monotonic):
    begin = clock()
    pose = list(checked(lambda: arm.get_position(is_radian=True), 'get_position'))
    pose_done = clock()
    joints = list(checked(lambda: arm.get_servo_angle(is_radian=True, is_real=True), 'get_servo_angle(actual)'))[:6]
    joint_done = clock()
    state = checked(arm.get_state, 'get_state')
    errors = list(checked(arm.get_err_warn_code, 'get_err_warn_code'))
    if len(pose) != 6 or len(joints) != 6 or not all(math.isfinite(float(x)) for x in pose+joints):
        raise ValueError('Invalid pose or joint feedback')
    return dict(host_monotonic_begin=begin, host_pose_received=pose_done,
                host_joints_received=joint_done, host_monotonic_end=clock(),
                pose_mm_rad=pose, position_sdk_m=[x/1000 for x in pose[:3]],
                joints_rad=joints, joint_query_is_real=True, state=state, error_warn=errors)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ip', default='192.168.1.197')
    p.add_argument('--duration', type=float, default=60)
    p.add_argument('--hz', type=float, default=20)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if not (math.isfinite(a.duration) and math.isfinite(a.hz) and a.duration > 0 and 0 < a.hz <= 50):
        p.error('duration must be positive; hz must be in (0,50]')
    if a.output is None:
        stamp = datetime.datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
        a.output = Path(__file__).resolve().parents[2]/'analysis_artifacts'/('xarm_readonly_'+stamp)
    from xarm.wrapper import XArmAPI
    a.output.mkdir(parents=True, exist_ok=False)
    arm = None
    rows = 0
    low, high = None, None
    start = None
    success = False
    failure = None
    try:
        arm = XArmAPI(a.ip, is_radian=True)
        # These properties come from SDK reporting; this script does not change them.
        metadata = dict(ip=a.ip, firmware=arm.version,
                        tcp_offset_sdk_mm_rad=arm.tcp_offset,
                        world_offset_sdk_mm_rad=arm.world_offset,
                        requested_hz=a.hz, requested_duration=a.duration,
                        joint_query_is_real=True,
                        utc_started=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        note='Raw SDK frame, NOT policy-ready. Sequential pose/joint queries are not synchronized. '
                             'Timestamps are host receive times, not hardware measurement timestamps. '
                             'Offsets are SDK cached reports; verify against Studio. '
                             'Joint query explicitly uses is_real=True (requires supported firmware). '
                             'TCP pose query is not independently verified encoder-based FK. No motion commands are sent.')
        (a.output/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        start = time.monotonic()
        next_print = start
        deadline = start
        with (a.output/'states.jsonl').open('x',encoding='utf-8',buffering=1) as f:
            while time.monotonic()-start < a.duration:
                row = sample(arm)
                row.update(index=rows, elapsed_s=row['host_monotonic_begin']-start)
                f.write(json.dumps(row)+'\n')
                xyz = row['position_sdk_m']
                low = xyz[:] if low is None else [min(x,y) for x,y in zip(low,xyz)]
                high = xyz[:] if high is None else [max(x,y) for x,y in zip(high,xyz)]
                rows += 1
                now = time.monotonic()
                if now >= next_print:
                    print('t={:.1f}s xyz_mm={} joints_rad={} state={} err/warn={}'.format(
                        now-start, [round(x,2) for x in row['pose_mm_rad'][:3]],
                        [round(x,4) for x in row['joints_rad']],row['state'],row['error_warn']),flush=True)
                    next_print = now+1
                deadline += 1/a.hz
                if deadline < now:
                    deadline = now
                time.sleep(max(0,deadline-time.monotonic()))
        success = True
    except KeyboardInterrupt:
        failure = 'interrupted by user; partial records preserved'
    except Exception as error:
        failure = '{}: {}'.format(type(error).__name__,error)
        raise
    finally:
        elapsed = time.monotonic()-start if start is not None else 0
        if arm is not None:
            arm.disconnect()
        summary = dict(complete=success, samples=rows, elapsed_s=elapsed,
                       observed_loop_hz=rows/elapsed if elapsed else 0,
                       position_range_mm=[1000*(h-l) for l,h in zip(low,high)] if low is not None else None,
                       failure=failure)
        (a.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print('Saved:',a.output,flush=True)
        print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
