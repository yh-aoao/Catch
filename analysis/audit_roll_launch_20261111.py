"""Robot-free launch audit using copies of the actual production MuJoCo model.

No reconstructed table: preserve compiled object, material, table and solver.
Disable robot collision/actuation and compensate robot gravity only; never
change target forces/material or push the object after launch.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import copy
import json
import sys
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'catch_it_copy'))
from gym_dcmm.envs.object_benchmark import make_object_benchmark,catalog
from ycb_objects.benchmark_geometry import bounds
OUT=ROOT/'analysis_artifacts/roll_launch_audit_20261111'
CAT=OUT/'catalog_used.json'

def passive_model(env):
    source=env.unwrapped.Dcmm.model
    model=copy.copy(source)
    bid=model.body('object').id
    obj=set(np.flatnonzero(model.geom_bodyid==bid).tolist())
    keep=obj|{model.geom('table_surface').id,model.geom('floor').id}
    for g in range(model.ngeom):
        if g not in keep:model.geom_contype[g]=model.geom_conaffinity[g]=0
    for p,(a,b) in enumerate(zip(model.pair_geom1,model.pair_geom2)):
        if int(a) not in keep or int(b) not in keep:model.pair_margin[p]=-1e6
    model.body_gravcomp[:]=1.;model.body_gravcomp[bid]=source.body_gravcomp[bid]
    model.actuator_gainprm[:]=0.;model.actuator_biasprm[:]=0.
    assert np.array_equal(model.body_inertia[bid],source.body_inertia[bid])
    assert np.array_equal(model.pair_friction,source.pair_friction)
    data=mujoco.MjData(model)
    return model,data

def trial(env,model,data,seed):
    env.reset(seed=seed)
    source=env.unwrapped.Dcmm
    bid=model.body('object').id
    qadr=int(model.jnt_qposadr[model.body_jntadr[bid]])
    vadr=int(model.jnt_dofadr[model.body_jntadr[bid]])
    mujoco.mj_resetData(model,data)
    data.qpos[:]=source.data.qpos;data.qvel[:]=0.
    data.qvel[vadr:vadr+6]=source.data.qvel[vadr:vadr+6]
    data.ctrl[:]=0.;data.qfrc_applied[:]=0.;data.xfrc_applied[:]=0.
    mujoco.mj_forward(model,data)
    table=model.geom('table_surface').id;floor=model.geom('floor').id
    table_center=data.geom_xpos[table].copy();size=model.geom_size[table].copy()
    edge=float(table_center[1]-size[1]);top=float(table_center[2]+size[2])
    np.testing.assert_allclose(table_center,source.data.geom_xpos[source.model.geom('table_surface').id],rtol=0,atol=1e-12)
    np.testing.assert_allclose(model.body_mass[bid],source.model.body_mass[bid],rtol=0,atol=1e-12)
    np.testing.assert_allclose(model.body_inertia[bid],source.model.body_inertia[bid],rtol=0,atol=1e-12)
    assert abs(edge-.9)<1e-12 and abs(top-.7)<1e-12
    shape=getattr(env.unwrapped,'benchmark_shape',None)
    radius=float(model.geom_size[env.unwrapped.object_id,0])
    initial=dict(position=data.qpos[qadr:qadr+3].tolist(),quaternion=data.qpos[qadr+3:qadr+7].tolist(),
                 velocity=data.qvel[vadr:vadr+6].tolist(),mass=float(model.body_mass[bid]),
                 inertia=model.body_inertia[bid].tolist())
    departed=None;speeds=[];stopped_at_2p5=None;on_table=False
    rot=np.zeros(9)
    for _ in range(int(5/model.opt.timestep)):
        mujoco.mj_step(model,data)
        assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
        position=data.qpos[qadr:qadr+3]
        mujoco.mju_quat2Mat(rot,data.qpos[qadr+3:qadr+7])
        high=bounds(shape,rot.reshape(3,3))[1] if shape is not None else np.full(3,radius)
        on_table=any(table in (c.geom1,c.geom2) for c in data.contact)
        on_floor=any(floor in (c.geom1,c.geom2) for c in data.contact)
        if position[1]+high[1]<edge and not on_table and not on_floor:
            departed=float(data.time);break
        if 2.49<data.time<2.51:stopped_at_2p5=bool(on_table and np.linalg.norm(data.qvel[vadr:vadr+3])<.02)
        if data.time>4.8:speeds.append(float(np.linalg.norm(data.qvel[vadr:vadr+3])))
    final_speed=float(np.mean(speeds)) if speeds else float(np.linalg.norm(data.qvel[vadr:vadr+3]))
    assert np.all(data.qfrc_actuator[vadr:vadr+6]==0)
    return dict(seed=seed,initial=initial,departure=departed,
        launch_valid=bool(departed is not None and departed<=2.5),
        arrival_window=bool(departed is not None and .8<=departed<=2.),
        early=bool(departed is not None and departed<.8),
        stopped=bool(departed is None and on_table and final_speed<.02),
        stopped_at_2p5=stopped_at_2p5,final_speed=final_speed,
        final_position=data.qpos[qadr:qadr+3].tolist(),
        table=dict(center=table_center.tolist(),half_size=size.tolist(),edge_y=edge,top_z=top))

def run(job):
    name,mode=job;rows=[];instances=[]
    for slot in range(8 if mode=='dr' else 1):
        env=make_object_benchmark(object_benchmark='roll_v1',benchmark_catalog=CAT,
            benchmark_options={'domain_randomization':'physical_dr_v1' if mode=='dr' else 'off'},
            object_motion='roll',task='Catching',test_object=name,viewer=False,render_mode=None,
            object_eval=False,benchmark_slot=0,benchmark_num_envs=1,benchmark_seed=62000+slot)
        try:
            # Runtime reset moves the table body; never clone the XML init pose.
            env.reset(seed=70000)
            model,data=passive_model(env)
            draw=copy.deepcopy(getattr(env,'benchmark_randomization',{})) if mode=='dr' else {}
            instances.append(dict(slot=slot,draw=draw))
            seeds=range(70000+slot*10,70000+(slot+1)*10) if mode=='dr' else range(70000,70080)
            for seed in seeds:
                r=trial(env,model,data,seed);r.update(slot=slot,draw=draw);rows.append(r)
        finally:env.close()
    times=[r['departure'] for r in rows if r['departure'] is not None and r['departure']<=2.5]
    result=dict(object=name,mode=mode,episodes=len(rows),valid=sum(r['launch_valid'] for r in rows),
        stopped=sum(r['stopped'] for r in rows),late=sum(r['departure'] is not None and r['departure']>2.5 for r in rows),
        not_departed=sum(r['departure'] is None for r in rows),early=sum(r['early'] for r in rows),
        arrival_window=sum(r['arrival_window'] for r in rows),time_quantiles=np.quantile(times,[0,.1,.5,.9,1]).tolist() if times else [],instances=instances,rows=rows)
    (OUT/f'{name}_{mode}.json').write_text(json.dumps(result,indent=2))
    return {k:v for k,v in result.items() if k not in ('rows','instances')}

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    jobs=[(n,m) for n in catalog(CAT)['roll'] for m in ('fixed','dr')]
    results=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(run,j) for j in jobs]):
            r=f.result();results.append(r);print(json.dumps(r),flush=True)
            (OUT/'summary.json').write_text(json.dumps(results,indent=2))
