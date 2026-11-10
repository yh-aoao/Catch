"""Passive fruits: separate late departure from actually settling on the table."""
import importlib.util
import json
from pathlib import Path
import sys
import copy
import mujoco
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'catch_it_copy'))
spec=importlib.util.spec_from_file_location('passive',ROOT/'analysis/validate_object_benchmark_20261103.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
from ycb_objects.benchmark_randomization import sample,material,scaled_geometry

def trial(name,profile,seed,dr=False,zero_rolling=False):
    shape=base.geometry(name,profile['kind']);p=copy.deepcopy(profile)
    draw=sample('roll',np.random.RandomState(100000+seed)) if dr else None
    if draw:shape=scaled_geometry(shape,draw);p=material(p,draw)
    if zero_rolling:p['rolling_length_m']=0.
    model,data=base.scene(shape,p,'roll',.002);rng=np.random.RandomState(seed)
    state=base.roll_state(shape,rng.uniform(*p['speed_mps']),np.deg2rad(rng.uniform(-10,10)),rng.uniform(-np.pi,np.pi))
    position=np.array([rng.uniform(-.15,.15),rng.uniform(*p['distance_m']),.7+state['clearance']+.0003])
    data.qpos[:]=np.r_[position,state['quat']];data.qvel[:]=state['velocity'];mujoco.mj_forward(model,data)
    bid=model.body('object').id;table=model.geom('table_surface').id;departure=None;speeds=[]
    for _ in range(2500):
        mujoco.mj_step(model,data)
        _,hi=base.bounds(shape,data.xmat[bid].reshape(3,3))
        if data.qpos[1]+hi[1]<.8:departure=float(data.time);break
        if data.time>4.8:speeds.append(float(np.linalg.norm(data.qvel[:3])))
    supported=any(table in (c.geom1,c.geom2) for c in data.contact)
    speed=float(np.mean(speeds)) if speeds else float(np.linalg.norm(data.qvel[:3]))
    return dict(object=name,seed=seed,dr=dr,draw=draw,zero_rolling=zero_rolling,
        speed_range=p['speed_mps'],departure=departure,left_by_2p5=departure is not None and departure<=2.5,
        left_by_5=departure is not None,settled_on_table=bool(departure is None and supported and speed<.02),
        final_speed=speed,final_position=data.qpos[:3].tolist(),initial_position=position.tolist(),
        initial_velocity=state['velocity'].tolist())

if __name__=='__main__':
    d=base.catalog()['roll'];rows=[];groups=[]
    for name,dr,old in [('apple',False,True),('apple',False,False),('apple',True,False),('lemon',False,False),('lemon',True,False)]:
        p=copy.deepcopy(d[name])
        if old:p['speed_mps']=[1.6,1.9]
        a=[trial(name,p,s,dr) for s in range(50)];rows+=a
        controls=[trial(name,p,r['seed'],dr,True) for r in a if r['settled_on_table']]
        g=dict(object=name,dr=dr,old_speed=old,speed=p['speed_mps'],trials=50,
            left_by_2p5=sum(r['left_by_2p5'] for r in a),left_by_5=sum(r['left_by_5'] for r in a),
            settled_on_table=sum(r['settled_on_table'] for r in a),zero_rolling_controls=controls)
        groups.append(g);print(json.dumps(g),flush=True)
    (ROOT/'analysis_artifacts/apple_speed_stop_check_20261110/results.json').write_text(json.dumps(dict(groups=groups,rows=rows),indent=2))
