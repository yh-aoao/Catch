"""Distance/pose exploration, followed by held-out passive validation."""
import importlib.util
from pathlib import Path
import copy
import json
import sys
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'catch_it_copy'))
spec=importlib.util.spec_from_file_location('stopcheck',ROOT/'analysis/check_fruit_stopping_20261110.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
OUT=ROOT/'analysis_artifacts/fruit_joint_launch_20261110'

def trial(name,p,seed,dr=True,yaw_degrees=(-180,180)):
    p=copy.deepcopy(p);shape=base.base.geometry(name,p['kind'])
    draw=base.sample('roll',np.random.RandomState(100000+seed)) if dr else None
    if draw:shape=base.scaled_geometry(shape,draw);p=base.material(p,draw)
    model,data=base.base.scene(shape,p,'roll',.002);rng=np.random.RandomState(seed)
    speed=rng.uniform(*p['speed_mps']);heading=np.deg2rad(rng.uniform(-10,10));yaw=np.deg2rad(rng.uniform(*yaw_degrees))
    state=base.base.roll_state(shape,speed,heading,yaw)
    pos=np.array([rng.uniform(-.15,.15),rng.uniform(*p['distance_m']),.7+state['clearance']+.0003])
    data.qpos[:]=np.r_[pos,state['quat']];data.qvel[:]=state['velocity'];mujoco.mj_forward(model,data)
    bid=model.body('object').id;depart=None;velocities=[]
    for _ in range(1250):
        mujoco.mj_step(model,data)
        _,hi=base.base.bounds(shape,data.xmat[bid].reshape(3,3))
        if data.qpos[1]+hi[1]<.8:depart=float(data.time);break
        if data.time>2.3:velocities.append(float(np.linalg.norm(data.qvel[:3])))
    return dict(seed=seed,departure=depart,stopped=bool(depart is None and np.mean(velocities)<.02),
        initial_speed=speed,yaw_degrees=float(np.rad2deg(yaw)),distance=float(pos[1]),draw=draw)

def summarize(name,distance,yaw,rows,dr):
    times=[r['departure'] for r in rows if r['departure'] is not None]
    return dict(object=name,distance=distance,yaw_degrees=yaw,dr=dr,trials=len(rows),
        departed=len(times),stopped=sum(r['stopped'] for r in rows),
        time_quantiles=np.quantile(times,[0,.1,.5,.9,1]).tolist() if times else [],
        early_departures=sum(t<.8 for t in times),rows=rows)

def run_candidate(job):
    name,distance,yaw=job
    p=copy.deepcopy(base.base.catalog()['roll'][name]);p['distance_m']=distance
    return summarize(name,distance,yaw,[trial(name,p,s,True,yaw) for s in range(20)],True)

if __name__=='__main__':
    from concurrent.futures import ProcessPoolExecutor,as_completed
    profiles=base.base.catalog()['roll'];results=json.loads((OUT/'scan.json').read_text()) if (OUT/'scan.json').exists() else []
    # Paired seeds across candidates, all DR enabled during exploration.
    jobs=[(name,distance,yaw) for name in ('apple','lemon')
        for distance in ([2.,2.2],[1.9,2.1],[1.8,2.0])
        for yaw in ([-180,180],[-180,-90],[-90,0],[0,90],[90,180])
        if not any(r['object']==name and r['distance']==distance and r['yaw_degrees']==yaw for r in results)]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(run_candidate,j) for j in jobs]):
            result=f.result();results.append(result)
            print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
            (OUT/'scan.json').write_text(json.dumps(results,indent=2))
