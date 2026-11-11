import sys,copy,json
from pathlib import Path
sys.path.insert(0,'/home/isee/Catch/catch_it_copy')
import numpy as np,mujoco
from gym_dcmm.envs.object_benchmark import make_object_benchmark
from ycb_objects import benchmark_randomization as br
OUT=Path(__file__).resolve().parent
rows=[]
for motion,names in [('roll',['original','tennis_ball','baseball','orange','apple','lemon','tomato_soup_can']),('bounce',['original','tennis_ball','baseball','racquetball','golf_ball','softball'])]:
 for name in names:
  kwargs=dict(object_benchmark=motion+'_v1',benchmark_catalog='/home/isee/Catch/catch_it_copy/ycb_objects/calibration_data/object_benchmark_all_targets_v1.json',object_motion=motion,task='Catching',test_object=name,viewer=False,render_mode=None,object_eval=False,benchmark_slot=0,benchmark_num_envs=1,benchmark_seed=123,bounce_grasp_version='controlled_hold_v1')
  env=make_object_benchmark(benchmark_options={'domain_randomization':'physical_dr_episode_v1'},**kwargs);env.reset(seed=9200);draw=env.benchmark_randomization.copy()
  original=br.sample;br.sample=lambda *a,**kw:dict(draw,version='physical_dr_v1',sampling='environment_creation')
  try:ref=make_object_benchmark(benchmark_options={'domain_randomization':'physical_dr_v1'},**kwargs)
  finally:br.sample=original
  try:
   ref.reset(seed=9200);a,b=env.unwrapped.Dcmm,ref.unwrapped.Dcmm
   ai=a.model.body('object').id;bi=b.model.body('object').id
   for key in ('body_mass','body_inertia','body_ipos'):np.testing.assert_allclose(getattr(a.model,key)[ai],getattr(b.model,key)[bi],rtol=2e-5,atol=2e-7)
   ag=np.flatnonzero(a.model.geom_bodyid==ai);bg=np.flatnonzero(b.model.geom_bodyid==bi)
   # Legacy physical_v2 changes analytic radius at reset but leaves its cached
   # AABB nominal. Compare against the correct freshly compiled primitive bound.
   for g in bg:
    if b.model.geom_type[g]==mujoco.mjtGeom.mjGEOM_SPHERE:
     primitive=mujoco.MjModel.from_xml_string(f'<mujoco><worldbody><geom type="sphere" size="{b.model.geom_size[g,0]}"/></worldbody></mujoco>')
     b.model.geom_aabb[g]=primitive.geom_aabb[0]
   errors={}
   for key in ('geom_size','geom_rbound','geom_pos','geom_aabb','geom_quat','geom_friction'):
    x=getattr(a.model,key)[ag];y=getattr(b.model,key)[bg];errors[key]=float(np.max(np.abs(x-y)));np.testing.assert_allclose(x,y,rtol=2e-4,atol=2e-6,err_msg=motion+name+key)
   for ga,gb in zip(ag,bg):
    if a.model.geom_type[ga]!=mujoco.mjtGeom.mjGEOM_MESH:continue
    ma=a.model.geom_dataid[ga];mb=b.model.geom_dataid[gb]
    va=a.model.mesh_vert[a.model.mesh_vertadr[ma]:a.model.mesh_vertadr[ma]+a.model.mesh_vertnum[ma]];vb=b.model.mesh_vert[b.model.mesh_vertadr[mb]:b.model.mesh_vertadr[mb]+b.model.mesh_vertnum[mb]]
    np.testing.assert_allclose(va,vb,rtol=2e-4,atol=2e-6)
   # Identical physical state, robot collision disabled: compare passive collision dynamics.
   for bot in (a,b):
    ids=set(np.flatnonzero(bot.model.geom_bodyid==bot.model.body('object').id));keep=ids|{bot.model.geom('floor').id,bot.model.geom('table_surface').id}
    for g in range(bot.model.ngeom):
     if g not in keep:bot.model.geom_contype[g]=bot.model.geom_conaffinity[g]=0
    bot.model.actuator_gainprm[:]=0;bot.model.actuator_biasprm[:]=0
   b.data.qpos[:]=a.data.qpos;b.data.qvel[:]=a.data.qvel
   for bot in (a,b):
    bot.data.qacc_warmstart[:]=0;bot.data.ctrl[:]=0;mujoco.mj_forward(bot.model,bot.data)
   for _ in range(100 if motion=='roll' else 1000):
    mujoco.mj_step(a.model,a.data);mujoco.mj_step(b.model,b.data)
   delta=float(np.linalg.norm(a.data.body('object').xpos-b.data.body('object').xpos))
   assert delta<1e-5,(motion,name,delta)
   rows.append(dict(motion=motion,object=name,geometry_errors=errors,passive_position_error_m=delta));print(motion,name,'PASS',delta,flush=True)
  finally:env.close();ref.close()
(OUT/'geometry_comparison_short_horizon.json').write_text(json.dumps(rows,indent=2))
