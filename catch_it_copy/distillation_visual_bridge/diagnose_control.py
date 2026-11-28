"""Matched-seed interventions for visual Basket: diagnostic only, not deployment."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import time
import numpy as np

MODES=('teacher','student','teacher-carry','teacher-base','teacher-arm-hand')


def intervention(mode,student,teacher,throw_ready):
    if mode=='teacher':return teacher.copy()
    if mode=='student':return student.copy()
    if mode=='teacher-carry':return student.copy() if throw_ready else teacher.copy()
    action=student.copy()
    if mode=='teacher-base':action[:2]=teacher[:2]
    elif mode=='teacher-arm-hand':action[2:]=teacher[2:]
    else:raise ValueError('Unknown intervention')
    return action


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--teacher-checkpoint',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--egl-device-id',type=int,default=0);p.add_argument('--episodes',type=int,default=5)
    p.add_argument('--seed',type=int,default=900123);p.add_argument('--modes',nargs='+',choices=MODES,default=list(MODES))
    args=p.parse_args()
    if args.episodes<1:p.error('episodes must be positive')
    if args.output.exists():p.error('Choose a new output directory')
    os.environ['MUJOCO_GL']='egl';os.environ['MUJOCO_EGL_DEVICE_ID']=str(args.egl_device_id)
    os.environ.setdefault('MPLCONFIGDIR','/tmp/catch_mpl')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    import torch
    torch.set_num_threads(1)
    from .runtime import validate_torch_device
    args.device=str(validate_torch_device(args.device))
    from .core import Student,SCHEMA,camera_specs,encode_depth
    from .teachers import FrozenTeacher,sha256
    from .factories import make_basket
    from .environments import DepthCamera
    from .run import student_step
    from distillation.core import ActionAdapter
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if saved.get('schema')!=SCHEMA:p.error('Expected visual Bridge student')
    cfg=copy.deepcopy(saved['config']);task=cfg['tasks']['basket']
    if args.teacher_checkpoint:task['checkpoint']=str(args.teacher_checkpoint.resolve())
    teacher=FrozenTeacher('basket',task,args.device);student=Student(cfg).to(args.device).eval()
    student.load_state_dict(saved['student'],strict=True);adapter=ActionAdapter(task)
    vision=dict(cfg['vision'],gl_backend='egl');specs=camera_specs(vision)
    args.output.mkdir(parents=True)
    (args.output/'audit.json').write_text(json.dumps(dict(student_sha256=sha256(args.checkpoint),
        teacher_sha256=teacher.hash,iteration=saved['iteration'],ablation=cfg.get('ablation','full'),
        config=cfg,modes=args.modes,episodes=args.episodes,seed=args.seed,
        note='Oracle teacher interventions use privileged state for causal diagnostics only; not student success rates.'),indent=2)+'\n')
    summaries={}
    for mode in args.modes:
        rows=[];env=None;sensor=None
        try:
            env=make_basket(**task['env_kwargs']);sensor=DepthCamera(env,dict(vision,camera=specs[0]['name']))
            for episode in range(args.episodes):
                seed=args.seed+episode;np.random.seed(seed);random.seed(seed)
                obs,_=env.reset(seed=seed);previous=np.zeros((1,20),np.float32);hidden=None
                errors=[];initial=copy.deepcopy(env.reset_stats);started=time.monotonic()
                trace=args.output/('{}_{:03d}.jsonl'.format(mode,episode))
                with trace.open('w') as stream:
                    for step in range(1,cfg['max_episode_steps']+1):
                        image=np.concatenate([encode_depth(sensor.capture_metres(camera=s['name']),s['near'],s['far']) for s in specs],axis=0)
                        packet=dict(depth=image[None],proprio=obs[None,:36],context=np.zeros((1,10),np.float32));packet['context'][0,0]=1.;packet['teacher_goal']=obs[42:45].copy()
                        policy,hidden=student_step(student,packet,previous,hidden)
                        target=adapter.canonical(teacher.predict(obs[None]))[0]
                        policy=adapter.executable(policy)[0];ready=bool(env._throw_ready)
                        action=intervention(mode,policy,target,ready)
                        delta=np.abs(policy-target);errors.append([float(delta[:2].mean()),float(delta[2:8].mean()),float(delta[8:].mean()),float(ready)])
                        local,relative,_,_=env._carry_measurements()
                        before=dict(step=step,time=float(env.Dcmm.data.time-env.start_time),throw_ready=ready,
                            student_action=policy.tolist(),teacher_action=target.tolist(),executed_action=action.tolist(),
                            ball_local=local.tolist(),ball_relative_velocity=relative.tolist(),
                            goal_distance=float(np.linalg.norm(env.basket_center[:2]-env.Dcmm.data.body('arm_base').xpos[:2])),
                            arm_target_error=(env.Dcmm.target_arm_qpos-env.Dcmm.data.qpos[15:21]).tolist())
                        obs,_,done,trunc,info=env.step(adapter.action_dict(action))
                        goal=student.extra.get('basket_position')
                        before['predicted_goal']=goal[0,0].detach().cpu().tolist() if goal is not None else None
                        before['actual_goal']=packet['teacher_goal'].tolist() if 'teacher_goal' in packet else None
                        before.update({k:info.get(k) for k in ('hand_contact','base_speed','released','release_stopped',
                            'purposeful_launch','reached_throw_phase','bridge_phase','terminated_reason','basket_region_hit')})
                        before['slip_speed']=info.get('carry_diagnostics',{}).get('slip_speed')
                        stream.write(json.dumps(before)+'\n');previous=env.last_action[None].copy()
                        if done or trunc:break
                    else:raise RuntimeError('Episode exceeded configured step limit')
                errors=np.asarray(errors)
                row=dict(episode=episode,seed=seed,steps=step,elapsed=info['elapsed'],
                    hit=bool(info['basket_region_hit']),strict_success=bool(info['safe_basket_success']),
                    reached_throw_phase=bool(info['reached_throw_phase']),reason=info['terminated_reason'],
                    purposeful_launch=bool(info.get('purposeful_launch',False)),initial_hold=initial,
                    action_mae_by_phase={phase:errors[errors[:,3]==value,:3].mean(0).tolist() if (errors[:,3]==value).any() else None
                        for phase,value in (('carry',0),('throw',1))},seconds=time.monotonic()-started)
                rows.append(row);print('[diagnose] '+mode+' '+json.dumps({k:v for k,v in row.items() if k!='initial_hold'}),flush=True)
                (args.output/(mode+'_episodes.json')).write_text(json.dumps(rows,indent=2)+'\n')
            from collections import Counter
            summaries[mode]=dict(episodes=len(rows),hits=sum(r['hit'] for r in rows),strict_successes=sum(r['strict_success'] for r in rows),
                reached_throw_phase=sum(r['reached_throw_phase'] for r in rows),reasons=dict(Counter(r['reason'] for r in rows)))
            (args.output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
        finally:
            if sensor is not None:sensor.close()
            if env is not None:env.close()
    print(json.dumps(summaries,indent=2),flush=True)

if __name__=='__main__':main()
