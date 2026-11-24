"""Inference-only collection from existing Catch teachers; no teacher updates."""
import argparse,copy,hashlib,importlib,json,os,random
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def collect(task,config,quota,episodes,seed,output):
    import torch,mujoco
    from distillation.teachers import Teacher,sha256
    from distillation.core import ActionAdapter,observation_array
    from gym_dcmm.utils.roll_rewards import hand_collision_ids
    torch.set_num_threads(1);os.chdir(ROOT);np.random.seed(seed);random.seed(seed)
    cfg=copy.deepcopy(config['tasks'][task]);cfg['checkpoint']=str((ROOT/cfg['checkpoint']).resolve())
    teacher=Teacher(cfg,'cpu');adapter=ActionAdapter(cfg)
    kwargs=dict(config.get('env_kwargs',{}),**cfg['env_kwargs'])
    kwargs.update(task='Catching',object_motion=task,viewer=False,render_mode=None,render_per_step=False,imshow_cam=False)
    if task in ('roll','bounce'):kwargs['test_object']='ycb:tennis_ball'
    module,name=cfg['env_factory'].split(':');env=getattr(importlib.import_module(module),name)(**kwargs)
    env.render=lambda:np.empty(0,np.float32)
    records=[];log=[]
    try:
      for episode in range(episodes):
        epseed=seed+episode*100003;np.random.seed(epseed%(2**32));random.seed(epseed)
        obs,_=env.reset();stable=0.;last_snapshot=-1.;snapshots=0
        for step in range(200):
            batch={k:({a:np.expand_dims(b,0) for a,b in v.items()} if isinstance(v,dict) else np.expand_dims(v,0)) for k,v in obs.items()}
            native=teacher.predict(observation_array(batch));command=adapter.action_dict(adapter.canonical(native))
            obs,_,term,trunc,info=env.step({k:v[0] for k,v in command.items()})
            m,d=env.Dcmm.model,env.Dcmm.data;hands=set(hand_collision_ids(m,env.hand_start_id).tolist())
            objects=set(np.flatnonzero(m.geom_bodyid==m.body('object').id).tolist())
            touch=any((int(c.geom1) in hands and int(c.geom2) in objects) or (int(c.geom2) in hands and int(c.geom1) in objects) for c in d.contact)
            palm=d.body('link6');ball=d.body('object');hv=np.zeros(6);bv=np.zeros(6)
            mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,m.body('link6').id,hv,0)
            mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,m.body('object').id,bv,0)
            local=palm.xmat.reshape(3,3).T@(ball.xpos-palm.xpos)
            relative=palm.xmat.reshape(3,3).T@(bv[3:]-hv[3:]-np.cross(hv[:3],ball.xpos-palm.xpos))
            dt=env.steps_per_policy*m.opt.timestep
            # A real supported contact window, not the task's permissive reward/success flag.
            valid=touch and np.linalg.norm(relative)<.5 and np.linalg.norm(local)<.18 and ball.xpos[2]>.10 and not term
            stable=stable+dt if valid else 0.
            now=float(d.time)
            if stable>=.04 and now-last_snapshot>=.16 and snapshots<3:
                record=dict(arm=d.qpos[15:21].tolist(),hand=d.qpos[21:37].tolist(),local=local.tolist(),
                    task=task,source=f'real_{task}_seed{epseed}_step{step}',episode=episode,seed=epseed,sim_time=now,
                    relative_velocity=relative.tolist(),arm_velocity=d.qvel[14:20].tolist(),
                    object_radius=float(m.geom_size[m.geom('object').id,0]),object_mass=float(m.body_mass[m.body('object').id]),
                    teacher_checkpoint=cfg['checkpoint'],teacher_sha256=teacher.hash,
                    supported_seconds=stable,contact_validated=True)
                records.append(record);last_snapshot=now;snapshots+=1
                print(task,'records',len(records),'episode',episode,flush=True)
                if len(records)>=quota:break
            if term or trunc:break
        log.append(dict(episode=episode,seed=epseed,snapshots=snapshots,steps=step+1))
        if (episode+1)%5==0:print(task,'episodes',episode+1,'records',len(records),flush=True)
        if len(records)>=quota:break
    finally:env.close()
    (output/(task+'_poses.json')).write_text(json.dumps(records,indent=2))
    (output/(task+'_collection.json')).write_text(json.dumps(dict(task=task,requested=quota,collected=len(records),episodes=log,teacher_sha256=teacher.hash),indent=2))
    return records

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=Path(__file__).parent/'configs/catch_sources.json')
 p.add_argument('--output',type=Path,required=True);p.add_argument('--per-task',type=int,default=30);p.add_argument('--max-episodes',type=int,default=120);p.add_argument('--seed',type=int,default=72100)
 p.add_argument('--tasks',nargs='+',choices=['throw','roll','bounce'],default=['throw','roll','bounce'])
 a=p.parse_args();os.chdir(ROOT)
 if min(a.per_task,a.max_episodes)<1:p.error('Positive quotas required')
 for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
 os.environ.setdefault('MPLCONFIGDIR','/tmp/catch_mpl')
 a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False);cfg=json.loads(a.config.read_text());records=[]
 for i,task in enumerate(a.tasks):records+=collect(task,cfg,a.per_task,a.max_episodes,a.seed+i*1009,a.output)
 if any(sum(r['task']==t for r in records)<a.per_task for t in a.tasks):raise RuntimeError('Incomplete pose coverage; inspect collection logs, no synthetic fallback')
 (a.output/'caught_poses.json').write_text(json.dumps(records,indent=2))
if __name__=='__main__':main()
