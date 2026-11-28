"""State-only teacher control/scoring audit in the exact distillation Basket scene."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).parent/'configs/catch.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--episodes',type=int,default=20);parser.add_argument('--seed',type=int,default=900123)
    args=parser.parse_args()
    if args.episodes<1:parser.error('episodes must be positive')
    if args.output.exists():parser.error('Choose a new output directory')
    # No visual policy is evaluated here: no renderer/context is required.
    os.environ['MUJOCO_GL']='disable'
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    import torch
    torch.set_num_threads(1)
    from distillation.core import ActionAdapter
    from .factories import make_basket
    from .teachers import FrozenTeacher
    from gym_dcmm.utils.basket_catching import hoop_crossing
    cfg=json.loads(args.config.read_text())['tasks']['basket']
    teacher=FrozenTeacher('basket',cfg,'cpu');adapter=ActionAdapter(cfg)
    env=make_basket(**cfg['env_kwargs']);rows=[]
    args.output.mkdir(parents=True)
    events=[];original=env._basket_check_flight
    def check(previous):
        original(previous)
        crossed,hit,radial=hoop_crossing(previous,env.Dcmm.data.qpos[37:40],env.basket_center,
            env.basket_cfg.basket_tilt_deg,env.basket_cfg.basket_radius,env.radius)
        if crossed:
            release=env._release or {}
            events.append(dict(hit=bool(hit),radial=float(radial),confirmed=bool(release.get('confirmed',False)),
                stopped=bool(release.get('valid_stop',False)),bridge_ready=bool(env._throw_ready)))
    env._basket_check_flight=check
    try:
        for episode in range(args.episodes):
            seed=args.seed+episode;np.random.seed(seed);random.seed(seed);events.clear()
            obs,_=env.reset(seed=seed);steps=0;error=0.;throw_frames=0;last_phase=float(obs[-1])
            for steps in range(1,1001):
                if obs.shape!=(66,):raise ValueError('Teacher observation changed')
                action=teacher.predict(obs[None])[0]
                obs,_,done,trunc,info=env.step(adapter.action_dict(action))
                error=max(error,float(np.max(np.abs(env.last_action-action))))
                throw_frames+=int(float(obs[-1])<.5);last_phase=float(obs[-1])
                if done or trunc:break
            else:raise RuntimeError('Episode did not finish')
            row=dict(episode=episode,seed=seed,success=bool(info['safe_basket_success']),
                reason=info['terminated_reason'],steps=steps,elapsed=info['elapsed'],
                reached_throw_phase=bool(info['reached_throw_phase']),released=bool(info['released']),
                release_stopped=bool(info['release_stopped']),action_conversion_max_error=error,
                crossing_events=copy.deepcopy(events),throw_frames=throw_frames,
                reset_attempts=info.get('reset_attempts'))
            rows.append(row)
            print(json.dumps(row),flush=True)
            (args.output/'records.json').write_text(json.dumps(rows,indent=2)+'\n')
        from collections import Counter
        summary=dict(episodes=len(rows),successes=sum(r['success'] for r in rows),
            success_rate=sum(r['success'] for r in rows)/len(rows),
            reached_throw_phase=sum(r['reached_throw_phase'] for r in rows),
            release_stopped=sum(r['release_stopped'] for r in rows),
            circle_hits_ignoring_stop_and_bridge=sum(any(e['hit'] for e in r['crossing_events']) for r in rows),
            reasons=dict(Counter(r['reason'] for r in rows)),
            action_conversion_max_error=max(r['action_conversion_max_error'] for r in rows),
            teacher_sha256=teacher.hash,basket_diameter_m=2*env.basket_cfg.basket_radius,
            autonomous_control=bool(env.basket_autonomous_control),integrated_joint_targets=bool(env.basket_integrated_arm_velocity),
            note='Circle hits are geometric diagnostics, NOT strict task successes. No student or depth policy is evaluated.')
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
    finally:env.close()

if __name__=='__main__':main()
