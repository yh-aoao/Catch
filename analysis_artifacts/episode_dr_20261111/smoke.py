import sys,json,time
from pathlib import Path
sys.path.insert(0,'/home/isee/Catch/catch_it_copy')
import numpy as np,mujoco
from gym_dcmm.envs.object_benchmark import make_object_benchmark
OUT=Path(__file__).resolve().parent
rows=[]
for motion,names in [('roll',['original','tennis_ball','baseball','orange','apple','lemon','tomato_soup_can']),('bounce',['original','tennis_ball','baseball','racquetball','golf_ball','softball'])]:
 for name in names:
  env=make_object_benchmark(object_benchmark=motion+'_v1',benchmark_catalog='/home/isee/Catch/catch_it_copy/ycb_objects/calibration_data/object_benchmark_all_targets_v1.json',benchmark_options={'domain_randomization':'physical_dr_episode_v1'},object_motion=motion,task='Catching',test_object=name,viewer=False,render_mode=None,object_eval=False,benchmark_slot=0,benchmark_num_envs=1,benchmark_seed=123,bounce_grasp_version='controlled_hold_v1')
  try:
   draws=[];times=[]
   for seed in (8100,8101,8100):
    start=time.monotonic();obs,info=env.reset(seed=seed);times.append(time.monotonic()-start)
    leaf=env.unwrapped;m=leaf.Dcmm.model;d=leaf.Dcmm.data;draw=env.benchmark_randomization.copy();draws.append(draw)
    np.testing.assert_allclose(m.body_mass[env.bid],env.mass*draw['mass_scale'])
    np.testing.assert_allclose(m.body_inertia[env.bid],env.inertia*draw['mass_scale']*draw['size_scale']**2)
    assert info['object_benchmark']['randomization']==draw
    if motion=='roll':
     from ycb_objects.benchmark_geometry import bounds
     shape=getattr(leaf,'benchmark_shape',None);low=bounds(shape,d.body('object').xmat.reshape(3,3))[0] if shape else np.full(3,-m.geom_size[leaf.object_id,0]);top=d.geom_xpos[m.geom('table_surface').id,2]+m.geom_size[m.geom('table_surface').id,2]
     np.testing.assert_allclose(d.body('object').xpos[2]+low[2]-top,.0003 if shape else .0002,atol=1e-9)
    for k in range(3):
     a={key:np.zeros(space.shape) for key,space in leaf.action_space.spaces.items()}
     obs,r,term,trunc,inf=env.step(a);assert np.isfinite(r) and np.isfinite(d.qpos).all();assert inf['object_benchmark']['randomization']==draw
   assert draws[0]!=draws[1] and draws[0]==draws[2]
   rows.append(dict(motion=motion,object=name,draws=draws,reset_seconds=times));print(motion,name,'PASS',times,flush=True)
  finally:env.close()
(OUT/'smoke.json').write_text(json.dumps(rows,indent=2))
