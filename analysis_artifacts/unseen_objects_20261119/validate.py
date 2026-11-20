import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]/'catch_it_copy';sys.path.insert(0,str(ROOT))
import numpy as np,mujoco
from gym_dcmm.envs.object_benchmark import make_object_benchmark
from ycb_objects.unseen_motion import sphere_drag
OUT=Path(__file__).resolve().parent;results=[]
for motion,name in [('bounce','table_tennis_ball'),('roll','peach')]:
 env=make_object_benchmark(object_benchmark=motion+'_v1',benchmark_catalog=str(ROOT/'ycb_objects/calibration_data/object_benchmark_unseen_v1.json'),benchmark_options={'domain_randomization':'physical_dr_v1'},object_motion=motion,task='Catching',test_object=name,viewer=False,render_mode=None,object_eval=False,roll_palm_version='palm_z_v1',bounce_grasp_version='controlled_hold_v1')
 try:
  e=env.unwrapped;m=e.Dcmm.model;bid=m.body('object').id;gid=m.geom('benchmark_collision').id
  physical=[]
  for seed in range(3):
   env.reset(seed=seed);physical.append(np.r_[m.body_mass[bid],m.body_inertia[bid],m.geom_size[gid]].copy())
   for _ in range(3):
    _,r,t,tr,_=env.step({'base':np.zeros(2),'arm':np.zeros(6),'hand':np.zeros(12)})
    assert np.isfinite(r)
    if t or tr:break
  assert all(np.array_equal(physical[0],r) for r in physical)
  trials=[]
  # Passive object motion; remove robot collision geometries only in this probe.
  ids=np.flatnonzero(m.geom_bodyid!=bid);surface=m.geom('floor' if motion=='bounce' else 'table_surface').id
  ids=ids[ids!=surface];m.geom_contype[ids]=0;m.geom_conaffinity[ids]=0
  for seed in range(20):
   env.reset(seed=seed);d=mujoco.MjData(m);mujoco.mj_copyData(d,m,e.Dcmm.data)
   j=int(m.body_jntadr[bid]);q=int(m.jnt_qposadr[j]);v=int(m.jnt_dofadr[j]);initial=d.qpos.copy();radius=float(m.geom_size[gid,0]);hit=False;peak=0.;apex_y=None;crossed=False
   front=d.geom_xpos[surface,1]-m.geom_size[surface,1] if motion=='roll' else None
   for step in range(int(2.5/m.opt.timestep)):
    d.ctrl[:]=0;d.qpos[:q]=initial[:q];d.qvel[:v]=0
    if motion=='bounce':d.qfrc_applied[v:v+3]=sphere_drag(d.qvel[v:v+3],radius)
    mujoco.mj_step(m,d)
    assert np.isfinite(d.qpos).all()
    if motion=='roll':
     if d.xpos[bid,1]<front-e.benchmark_shape['radius']:crossed=True;break
    else:
     contacts=np.asarray(d.contact.geom).reshape(-1,2)
     if any(surface in pair and gid in pair for pair in contacts):hit=True
     clearance=float(d.xpos[bid,2]-radius-d.geom_xpos[surface,2])
     if hit and clearance>peak:peak=clearance;apex_y=float(d.xpos[bid,1])
     if hit and peak>.1 and d.qvel[v+2]<0:break
   trials.append(dict(seed=seed,crossed=crossed,hit=hit,apex_clearance=peak,apex_y=apex_y))
  row=dict(motion=motion,object=name,physics=physical[0].tolist(),trials=trials)
  results.append(row);(OUT/'physics_checks.json').write_text(json.dumps(results,indent=2))
  if motion=='roll':assert all(r['crossed'] for r in trials),trials
  else:assert all(r['hit'] and .4<r['apex_clearance']<1.1 and -.1<r['apex_y']<1.4 for r in trials),trials
  print(name,'PASS',flush=True)
 finally:env.close()
