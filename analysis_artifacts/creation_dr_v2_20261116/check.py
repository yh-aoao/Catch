from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'catch_it_copy'))
import numpy as np,mujoco
from gym_dcmm.envs.object_benchmark import make_object_benchmark
ROOT=Path(__file__).resolve().parents[2]/'catch_it_copy'
CAT=ROOT/'ycb_objects/calibration_data/object_benchmark_creation_v2.json'
catalog=json.loads(CAT.read_text());out=[]
for motion in ('roll','bounce'):
 for name in catalog[motion]:
  env=make_object_benchmark(object_benchmark=motion+'_v1',benchmark_catalog=str(CAT),benchmark_options={'domain_randomization':'physical_dr_v1'},object_motion=motion,task='Catching',test_object=name,viewer=False,render_mode=None,object_eval=False,roll_palm_version='palm_z_v1',bounce_grasp_version='controlled_hold_v1')
  try:
   e=env.unwrapped;m=e.Dcmm.model;bid=m.body('object').id;gid=e.object_id
   snapshots=[];launches=[]
   for seed in (123,124,125):
    env.reset(seed=seed)
    snapshots.append(np.r_[m.body_mass[bid],m.body_inertia[bid],m.geom_size[gid]].copy())
    launches.append(np.r_[e.object_pos3d,e.object_vel6d].tolist())
    _,r,_,_,_=env.step({'base':np.zeros(2),'arm':np.zeros(6),'hand':np.zeros(12)})
    assert np.isfinite(r)
   assert all(np.array_equal(snapshots[0],s) for s in snapshots)
   assert launches[0]!=launches[1]
   out.append(dict(motion=motion,object=name,fixed_physics=snapshots[0].tolist(),launch_varies=True))
   if name=='small_cylinder':
    successes=0
    for seed in range(40):
     env.reset(seed=seed);d=mujoco.MjData(m);mujoco.mj_copyData(d,m,e.Dcmm.data)
     table=m.geom('table_surface').id;front=d.geom_xpos[table,1]-m.geom_size[table,1]
     j=int(m.body_jntadr[bid]);q=int(m.jnt_qposadr[j]);v=int(m.jnt_dofadr[j])
     initial_qpos=d.qpos.copy();initial_qvel=d.qvel.copy();elapsed=0.
     # Passive launch test: lock robot coordinates and disable its actuators.
     while elapsed<3.:
      d.ctrl[:]=0;d.qpos[:q]=initial_qpos[:q];d.qvel[:v]=0
      mujoco.mj_step(m,d);elapsed+=m.opt.timestep
      if d.xpos[bid,1]<front-e.benchmark_shape['radius']:
       successes+=1;break
    out[-1]['passive_front_crossings']=successes;out[-1]['passive_trials']=40
  finally:env.close()
Path(__file__).with_name('checks.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
assert successes==40,successes
