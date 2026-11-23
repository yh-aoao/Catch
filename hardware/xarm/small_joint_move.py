#!/usr/bin/env python3
"""Small xArm Mode-0 joint test. Read-only preview unless --execute is explicit.

No homing, reset, error clearing, mode switching, or automatic return motion.
"""
import argparse
import datetime
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]


def ok(result, name):
    code, value = result
    if code != 0:
        raise RuntimeError('{} failed: {}'.format(name,code))
    return value


def joints(arm):
    value=list(ok(arm.get_servo_angle(is_radian=False,is_real=True),'actual joints'))[:6]
    if len(value)!=6 or not all(math.isfinite(x) for x in value):
        raise RuntimeError('Invalid actual joint feedback')
    return value


def health(arm, idle=False):
    state=ok(arm.get_state(),'get_state')
    errors=ok(arm.get_err_warn_code(),'get_err_warn_code')
    if not arm.connected or list(errors)!=[0,0]:
        raise RuntimeError('Disconnected or error/warning: {}'.format(errors))
    if arm.mode != 0:
        raise RuntimeError('Requires ordinary position mode 0; current mode={}. Configure in Studio first.'.format(arm.mode))
    if state not in ((2,) if idle else (0,1,2)):
        raise RuntimeError('Unexpected robot state {}; requires ready state 2 before starting.'.format(state))
    return state


def target_from(current, joint, delta):
    if joint not in range(1,7) or not math.isfinite(delta) or not 0<abs(delta)<=2:
        raise ValueError('Joint must be 1..6; delta must be nonzero and at most 2 degrees')
    target=current[:];target[joint-1]+=delta
    return target


def check_limit(arm, target):
    limited=ok(arm.is_joint_limit(target,is_radian=False),'is_joint_limit')
    if limited is not False:
        raise RuntimeError('Controller joint-limit check failed or target is limited: {}'.format(limited))


def stop(arm, log):
    try:
        code=arm.set_state(4)
        log(dict(event='stop_requested',code=code))
    except BaseException as error:
        log(dict(event='stop_request_failed',error=str(error)))
        print('Software stop could not be confirmed; use the physical emergency stop.',flush=True)


def execute_move(arm, current, target, joint, speed, acceleration, log,
                 clock=time.monotonic, sleep=time.sleep):
    health(arm,idle=True)
    fresh=joints(arm)
    if max(abs(x-y) for x,y in zip(fresh,current))>.1:
        raise RuntimeError('Robot moved after preview; re-run from the current pose')
    check_limit(arm,target)
    started=clock();stable_since=None
    # Mark before transmission: an exception does not prove the command was not received.
    log(dict(event='command_attempt',joint=joint,target_deg=target,speed_deg_s=speed,acceleration_deg_s2=acceleration))
    try:
        code=arm.set_servo_angle(angle=target,is_radian=False,speed=speed,
                                 mvacc=acceleration,relative=False,wait=False,radius=-1)
        if code!=0:raise RuntimeError('Motion command returned {}'.format(code))
        while True:
            begin=clock();state=health(arm);actual=joints(arm);now=clock()
            log(dict(event='feedback',elapsed_s=now-started,joints_deg=actual,state=state,
                     query_duration_s=now-begin))
            if now-begin>.5:
                raise RuntimeError('Feedback request took over 0.5s')
            for i,(a,start,end) in enumerate(zip(actual,current,target)):
                margin=.25 if i==joint-1 else .15
                if not min(start,end)-margin<=a<=max(start,end)+margin:
                    raise RuntimeError('Joint {} left the small test envelope'.format(i+1))
            reached=max(abs(a-b) for a,b in zip(actual,target))<=.05 and state==2
            if reached:
                stable_since=now if stable_since is None else stable_since
                if now-stable_since>=.3:
                    log(dict(event='completed',final_joints_deg=actual));return actual
            else:stable_since=None
            if now-started>15:
                raise TimeoutError('Target not confirmed within 15s')
            sleep(max(0,.05-(clock()-begin)))
    except BaseException:
        stop(arm,log)
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ip',default='192.168.1.197')
    p.add_argument('--joint',type=int,default=1)
    p.add_argument('--delta-deg',type=float,default=.5)
    p.add_argument('--speed-deg-s',type=float,default=1)
    p.add_argument('--acc-deg-s2',type=float,default=2)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    try:target_from([0.]*6,args.joint,args.delta_deg)
    except ValueError as error:p.error(str(error))
    if not (math.isfinite(args.speed_deg_s) and 0<args.speed_deg_s<=2 and
            math.isfinite(args.acc_deg_s2) and 0<args.acc_deg_s2<=5):
        p.error('speed must be in (0,2] deg/s; acceleration in (0,5] deg/s^2')
    from xarm.wrapper import XArmAPI
    stamp=datetime.datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
    out=args.output or ROOT/'analysis_artifacts'/('xarm_small_move_'+stamp)
    out.mkdir(parents=True,exist_ok=False)
    arm=None
    with (out/'events.jsonl').open('x',buffering=1) as file:
        def log(value):file.write(json.dumps(dict(host_monotonic=time.monotonic(),**value))+'\n')
        try:
            arm=XArmAPI(args.ip,is_radian=False)
            arm.set_timeout(1)
            if arm.axis!=6:raise RuntimeError('Expected a six-axis xArm')
            health(arm,idle=True)
            current=joints(arm);time.sleep(.15);again=joints(arm)
            if max(abs(x-y) for x,y in zip(current,again))>.03:
                raise RuntimeError('Robot is not stationary')
            target=target_from(again,args.joint,args.delta_deg)
            check_limit(arm,target)
            preview=dict(event='preview',ip=args.ip,execute=args.execute,joint=args.joint,
                         start_deg=again,target_deg=target,delta_deg=args.delta_deg,
                         speed_deg_s=args.speed_deg_s,acceleration_deg_s2=args.acc_deg_s2,
                         tcp_offset_sdk=arm.tcp_offset,world_offset_sdk=arm.world_offset,
                         firmware=arm.version,note='Controller joint-limit check only; path collision/clearance not verified')
            log(preview);print(json.dumps(preview,indent=2),flush=True)
            if not args.execute:
                print('Preview only. No motion command sent. Add --execute to perform this small test.',flush=True)
                return
            print('Moving once; remains at the new pose. Do not issue Studio motion commands concurrently.',flush=True)
            final=execute_move(arm,again,target,args.joint,args.speed_deg_s,args.acc_deg_s2,log)
            print('Completed. Actual joints (deg):',final,flush=True)
        finally:
            if arm is not None:arm.disconnect()
            print('Log:',out,flush=True)


if __name__=='__main__':main()
