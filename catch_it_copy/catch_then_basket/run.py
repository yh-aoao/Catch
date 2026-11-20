"""Evaluate continuous catch → basket without resetting the physical scene."""
import argparse
import os
from pathlib import Path
import json
import hashlib


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=10)
    p.add_argument('--seed',type=int,default=20261119)
    p.add_argument('--device',default='cpu')
    p.add_argument('--viewer',action='store_true')
    p.add_argument('--egl-device-id',type=int,default=0)
    p.add_argument('--hold-seconds',type=float,default=.3)
    p.add_argument('--relative-speed',type=float,default=.3)
    p.add_argument('--palm-distance',type=float,default=.18)
    p.add_argument('--check-handoff',action='store_true',help='Check state preservation using a forced handoff; not a success evaluation')
    args=p.parse_args()
    if args.episodes<1:p.error('--episodes must be positive')
    checkpoint=args.checkpoint.resolve();output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    os.environ['MUJOCO_GL']='glfw' if args.viewer else 'egl'
    os.environ['MUJOCO_EGL_DEVICE_ID']=str(args.egl_device_id)
    root=Path(__file__).resolve().parents[1];os.chdir(root)
    import random
    import numpy as np
    import torch
    from .environment import CatchThenBasketEnv
    from distillation_four_task_future.core import Student,SCHEMA,robot_input,command_context,camera_specs,encode_depth
    from distillation_four_task_future.environments import DepthCamera
    from distillation.core import ActionAdapter
    torch.set_num_threads(1);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if ck['schema']!=SCHEMA:raise ValueError('Expected four-task Future student')
    cfg=ck['config'];student=Student(cfg).to(args.device);student.load_state_dict(ck['student'],strict=True);student.eval()
    kwargs={t:dict(cfg['env_kwargs'],**cfg['tasks'][t].get('env_kwargs',{})) for t in ('throw','basket')}
    env=CatchThenBasketEnv(kwargs['basket'],kwargs['throw'],args.hold_seconds,args.relative_speed,args.palm_distance)
    sensor=DepthCamera(env,cfg['vision']);specs=camera_specs(cfg['vision'])
    adapters={t:ActionAdapter(cfg['tasks'][t]) for t in ('throw','basket')}
    window=None;records=[]
    metadata=dict(checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  iteration=ck.get('iteration'),seed=args.seed,hold_seconds=args.hold_seconds,
                  relative_speed=args.relative_speed,palm_distance=args.palm_distance,
                  diagnostic_forced_handoff=args.check_handoff,hidden_reset_at_handoff=True,
                  physical_reset_at_handoff=False,catch_ball_physics='original_throw',basket_rehearsal=False)
    (output/'settings.json').write_text(json.dumps(metadata,indent=2))
    try:
        for episode in range(args.episodes):
            obs,_=env.reset(seed=args.seed+episode)
            if args.viewer and window is None:
                import mujoco.viewer
                window=mujoco.viewer.launch_passive(env.Dcmm.model,env.Dcmm.data)
            if args.check_handoff:
                obs=env.switch_to_basket()
                record=dict(episode=episode,transition=env.transition)
                # Exercise the freshly initialized controller and logical state too.
                env.step(adapters['basket'].action_dict(np.zeros(20,np.float32)))
                records.append(record);continue
            hidden=None;previous=np.zeros(20,np.float32);entered_throw=False;retention=0.;max_retention=0.
            for step in range(1000):
                task='throw' if env.chain_phase=='catch' else 'basket'
                depth=np.concatenate([encode_depth(sensor.capture_metres(s['name']),s['near'],s['far']) for s in specs],axis=0)
                prop=robot_input(obs,env.Dcmm.data.qpos[15:21],env.Dcmm.data.qvel[14:20])
                context=command_context(obs,task)
                def tensor(a):return torch.as_tensor(a,dtype=torch.float32,device=args.device)[None,None]
                with torch.no_grad():action,hidden,_=student(tensor(depth),tensor(prop),tensor(previous),tensor(context),hidden)
                previous=action[0,0].cpu().numpy()
                obs,_,terminated,truncated,info=env.step(adapters[task].action_dict(previous))
                if info.get('chain_switched'):hidden=None;previous=np.zeros(20,np.float32)
                if env.chain_phase=='basket':
                    entered_throw|=bool(env.basket_mobile_state.get('unlocked'))
                    retention=retention+env.steps_per_policy*env.Dcmm.model.opt.timestep if env.hold_measurements()['touching'] else 0.
                    max_retention=max(max_retention,retention)
                if window is not None:
                    window.sync()
                    if not window.is_running():raise KeyboardInterrupt
                if terminated or truncated:break
            record=dict(episode=episode,seed=args.seed+episode,steps=step+1,stable_catch=env.transition is not None,
                        retained_after_handoff=max_retention>=.3,entered_throw_phase=entered_throw,
                        success=bool(env.transition is not None and info.get('safe_basket_success',False)),
                        failure=info.get('chain_failure','step_limit'),max_stable_hold_seconds=env.max_hold_seconds,
                        max_post_handoff_contact_seconds=max_retention,transition=env.transition)
            records.append(record)
            with (output/'episodes.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps(record,ensure_ascii=False),flush=True)
        caught=sum(r.get('stable_catch',False) for r in records);success=sum(r.get('success',False) for r in records)
        result=dict(metadata=metadata,episodes=len(records),stable_catches=caught,successes=success,
                    stable_catch_rate=None if args.check_handoff else caught/len(records),end_to_end_success_rate=None if args.check_handoff else success/len(records),
                    success_given_stable_catch=success/caught if caught else None,records=records)
        (output/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
        print('Saved:',output/'summary.json',flush=True)
    finally:
        if window is not None:window.close()
        sensor.close();env.close()

if __name__=='__main__':main()
