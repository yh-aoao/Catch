"""Passive, robot-free motion validation of the exact benchmark geometry/pairs."""
import json,sys
from pathlib import Path
import mujoco
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'catch_it_copy'))
from ycb_objects.benchmark_geometry import geometry,inject,bounds,roll_state
from gym_dcmm.envs.object_benchmark import catalog,bounce_launch
OUT=Path(__file__).resolve().parents[1]/'analysis_artifacts/object_benchmark_20261103'

def scene(shape,p,motion,dt):
 surface=('<geom name="table_surface" type="box" pos="0 2.4 .65" size="1.2 1.6 .05"/>' if motion=='roll' else '')
 xml=f'<mujoco><option timestep="{dt}" integrator="RK4"/><worldbody><geom name="floor" type="plane" size="5 5 .1"/>{surface}<body name="object"><freejoint/><geom name="object" type="sphere" size=".04"/></body></worldbody></mujoco>'
 model=mujoco.MjModel.from_xml_string(inject(xml,shape,p,'table_surface' if motion=='roll' else 'floor'));return model,mujoco.MjData(model)

def trial(name,p,motion,seed,dt):
 s=geometry(name,p['kind'],p.get('inertia_factor',.4));model,data=scene(s,p,motion,dt);rng=np.random.RandomState(seed)
 if motion=='roll':
  state=roll_state(s,rng.uniform(*p.get("speed_mps",[.8,1.3])),np.deg2rad(rng.uniform(-10,10)),rng.uniform(-np.pi,np.pi));position=np.array([rng.uniform(-.15,.15),rng.uniform(*p.get("distance_m",[2.0,2.2])),.7+state['clearance']+.0003]);q=state['quat'];v=state['velocity'];plan={'no_slip_error':state['no_slip_error']}
 else:position,v,plan=bounce_launch(p,s['radius'],rng);q=np.array([1.,0,0,0])
 data.qpos[:]=np.r_[position,q];data.qvel[:]=v;mujoco.mj_forward(model,data)
 first=None;peak=None;depart=None;max_penetration=0.;max_speed=0.
 body=model.body('object').id;imat=np.zeros(9);mujoco.mju_quat2Mat(imat,model.body_iquat[body]);rI=imat.reshape(3,3);I=rI@np.diag(model.body_inertia[body])@rI.T
 def energy():return float(.5*s['mass']*np.dot(data.qvel[:3],data.qvel[:3])+.5*data.qvel[3:]@I@data.qvel[3:]+s['mass']*9.81*data.qpos[2])
 energy0=energy();max_energy=energy0
 for _ in range(int(2.5/dt)):
  mujoco.mj_step(model,data)
  assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
  max_speed=max(max_speed,float(np.linalg.norm(data.qvel[:3])));max_energy=max(max_energy,energy())
  ids=[]
  for c in data.contact:
   ids.extend([int(c.geom1),int(c.geom2)]);max_penetration=max(max_penetration,-float(c.dist))
  if motion=='bounce':
   if data.ncon and first is None:first=float(data.time)
   if first is not None:
    if peak is None or data.qpos[2]>peak['position'][2]:peak={'time':float(data.time),'position':data.qpos[:3].tolist()}
    if peak and data.time>peak['time']+.08:break
  else:
   R=data.xmat[body].reshape(3,3);low,high=bounds(s,R)
   if depart is None and data.qpos[1]+high[1]<.8:
    depart={'time':float(data.time),'position':data.qpos[:3].tolist()};break
 result=dict(object=name,motion=motion,seed=seed,timestep=dt,initial_position=position.tolist(),initial_velocity=v.tolist(),plan=plan,
             final_position=data.qpos[:3].tolist(),departure=depart,first_floor_contact=first,rebound_peak=peak,
             max_penetration_m=max_penetration,max_speed=max_speed,initial_energy_J=energy0,max_energy_J=max_energy,
             energy_gain_J=max_energy-energy0,finite=True)
 if motion=='roll':result['passed']=bool(depart and max_energy<=energy0+.02 and plan['no_slip_error']<1e-10)
 else:
  apex=peak['position'][2]-s['radius'] if peak else 0
  result['apex_clearance_m']=apex;result['apex_y']=peak['position'][1] if peak else None
  result['passed']=bool(first is not None and .35<apex<1.05 and peak and .15<peak['position'][1]<1.2 and max_energy<=energy0+.02)
 return result

if __name__=='__main__':
 d=catalog();rows=[]
 for motion in ('roll','bounce'):
  for name,p in d[motion].items():
   group=[trial(name,p,motion,seed,.002 if motion=='roll' else .000125) for seed in range(10)]
   rows.extend(group);print(motion,name,'passed',sum(r['passed'] for r in group),'/10',flush=True)
 fine=[]
 for motion in ('roll','bounce'):
  for name,p in d[motion].items():fine.append(trial(name,p,motion,0,.001 if motion=='roll' else .0000625))
 report={'assumptions':'Passive scene, exact same object collision/inertia/surface pair. No robot or post-release assistance. Priors not material measurements.','rows':rows,'half_timestep_seed0':fine}
 OUT.mkdir(exist_ok=True,parents=True);(OUT/'motion_validation.json').write_text(json.dumps(report,indent=2))
 print('Total',sum(r['passed'] for r in rows),'/',len(rows))
