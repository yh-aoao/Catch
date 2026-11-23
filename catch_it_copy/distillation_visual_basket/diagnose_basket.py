"""Matched-seed diagnostic: native/converted teacher and autonomous student.
Does not change training config or checkpoints; detailed traces are JSONL.
"""
import argparse,os,json,copy
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--teacher',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=5);p.add_argument('--seed',type=int,default=20261117)
    p.add_argument('--modes',nargs='+',default=['teacher_native','teacher_converted','student','student_native'])
    a=p.parse_args();checkpoint=a.checkpoint.resolve();teacherpath=a.teacher.resolve();output=a.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    os.environ['MUJOCO_GL']='egl';os.environ.setdefault('MUJOCO_EGL_DEVICE_ID','0')
    root=Path(__file__).resolve().parents[1];os.chdir(root)
    import random,numpy as np,torch
    torch.set_num_threads(1)
    from .core import Student,robot_input,command_context,camera_specs,encode_depth
    from .teachers import FrozenTeacher
    from .factories import AutonomousBasketEnv
    from .environments import DepthCamera,hand_contact
    from distillation.core import ActionAdapter
    from configs.env import DcmmCfg
    from gym_dcmm.utils.basket_grip import palm_coordinates
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True);config=saved['config']
    cfg=copy.deepcopy(config['tasks']['basket']);cfg['checkpoint']=str(teacherpath)
    teacher=FrozenTeacher('basket',cfg,'cpu');adapter=ActionAdapter(cfg)
    student=Student(config).eval();student.load_state_dict(saved['student'],strict=True)
    np.random.seed(123);random.seed(123)
    kw=dict(config['env_kwargs'],**cfg['env_kwargs']);kw.update(task='Catching',object_motion='throw_basket',viewer=False,render_mode=None,render_per_step=False)
    env=AutonomousBasketEnv(**kw);env.render=lambda:np.empty(0,np.float32)
    sensor=DepthCamera(env,config['vision']);specs=camera_specs(config['vision']);records=[]
    def batch(x):
        return {k:batch(v) if isinstance(v,dict) else np.asarray(v)[None] for k,v in x.items()}
    def plain(x):
        if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [plain(v) for v in x]
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        return x
    try:
        for mode in a.modes:
            if mode not in ('teacher_native','teacher_converted','student','student_native'):raise ValueError(mode)
            env.basket_autonomous_control=mode not in ('teacher_native','student_native')
            for ep in range(a.episodes):
                seed=a.seed+ep;np.random.seed(seed);random.seed(seed);torch.manual_seed(seed)
                obs,_=env.reset(seed=seed);previous=np.zeros(20,np.float32);hidden=None
                reached=False;touchsteps=0;minimum=1e9;maxspeed=0.;trace=[];reward_sum=0.
                for step in range(512):
                    reference=teacher.predict(batch(obs))[0]
                    phase=bool(env.basket_mobile_state['unlocked'])
                    if mode=='teacher_converted' and not phase:
                        dt=env.steps_per_policy*env.Dcmm.model.opt.timestep
                        reference[2:8]=np.clip((env.basket_carry_arm-env.Dcmm.data.qpos[15:21])/(np.asarray(DcmmCfg.basket_joint_speed_limits)*dt),-1,1)
                        indices=[0,2,3,4,6,7,8,10,11,13,14,15]
                        reference[8:]=np.clip((env.basket_carry_hand-env.Dcmm.target_hand_qpos)[indices]/(.15*DcmmCfg.basket_hand_action_gain),-1,1)
                    if mode.startswith('student'):
                        depth=np.concatenate([encode_depth(sensor.capture_metres(s['name']),s['near'],s['far']) for s in specs])
                        proprio=robot_input(obs,env.Dcmm.data.qpos[15:21],env.Dcmm.data.qvel[14:20])
                        context=command_context(obs,'basket')
                        def tensor(x):return torch.as_tensor(x)[None,None]
                        with torch.no_grad():act,hidden,_=student(tensor(depth),tensor(proprio),tensor(previous),tensor(context),hidden)
                        action=act[0,0].numpy()
                    else:action=reference
                    parking=env._basket_parking_state();touch=hand_contact(env)
                    minimum=min(minimum,parking['distance']);maxspeed=max(maxspeed,parking['speed']);touchsteps+=int(touch)
                    trace.append(dict(step=step,elapsed=float(env.Dcmm.data.time-env.start_time),unlocked=phase,touch=touch,
                        parking_distance=float(parking['distance']),parking_error=parking['error_world'].tolist(),base_speed=float(parking['speed']),
                        action=action.tolist(),teacher=reference.tolist(),ball_local=palm_coordinates(env.Dcmm.data,env.Dcmm.data.qpos[37:40]).tolist(),
                        arm_qpos=env.Dcmm.data.qpos[15:21].tolist(),basket_prediction=student.extra['basket_position'][0,0].detach().tolist() if mode.startswith('student') else None))
                    obs,reward,done,truncated,info=env.step(adapter.action_dict(action));reward_sum+=reward
                    previous=action.copy();reached|=bool(env.basket_mobile_state['unlocked'])
                    if done or truncated:break
                record=plain(dict(mode=mode,episode=ep,seed=seed,iteration=saved['iteration'],steps=step+1,
                    success=bool(info.get('safe_basket_success')),reason=info.get('terminated_reason'),parking_unlocked=reached,
                    min_parking_distance=minimum,max_base_speed=maxspeed,contact_fraction=touchsteps/(step+1),reward=reward_sum,
                    control=info.get('basket_control'),release_check=env.basket_throw_check))
                records.append(record)
                (output/(mode+'_%02d.json'%ep)).write_text(json.dumps(record,indent=2))
                with (output/(mode+'_%02d.jsonl'%ep)).open('w') as f:
                    for row in trace:f.write(json.dumps(row)+'\n')
                (output/'records.json').write_text(json.dumps(records,indent=2))
                print(json.dumps({k:record[k] for k in ('mode','episode','success','reason','parking_unlocked','steps','min_parking_distance','contact_fraction')}),flush=True)
        summary={}
        from collections import Counter
        for mode in a.modes:
            rows=[r for r in records if r['mode']==mode]
            summary[mode]=dict(episodes=len(rows),successes=sum(r['success'] for r in rows),parking_unlocked=sum(r['parking_unlocked'] for r in rows),reasons=dict(Counter(r['reason'] for r in rows)))
        (output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
    finally:sensor.close();env.close()

if __name__=='__main__':main()
