from pathlib import Path
import sys,json
sys.path.insert(0,'/home/isee/Catch/catch_it_copy')
import numpy as np,mujoco
from gym_dcmm.envs.object_benchmark import make_object_benchmark
from ycb_objects.benchmark_geometry import mesh_points
OUT=Path(__file__).resolve().parent;rows=[]
for motion in ('roll','bounce'):
 env=make_object_benchmark(object_benchmark=motion+'_v1',benchmark_catalog='/home/isee/Catch/catch_it_copy/ycb_objects/calibration_data/object_benchmark_all_targets_v1.json',benchmark_options={'domain_randomization':'off'},object_motion=motion,task='Catching',test_object='original',viewer=False,render_mode=None,object_eval=False,bounce_grasp_version='controlled_hold_v1')
 try:
  env.reset(seed=123);e=env.unwrapped;m,d=e.Dcmm.model,e.Dcmm.data;gid=e.hand_start_id
  q=d.qpos.copy();d.qpos[21:37]=0;mujoco.mj_forward(m,d)
  rotation=d.body('link6').xmat.reshape(3,3);origin=d.body('link6').xpos.copy()
  points=(mesh_points(m,d,gid)-origin)@rotation
  vals,vecs=np.linalg.eigh(np.cov(points.T));thin=vecs[:,0]
  tip=m.geom('fingertip_').id;before=(d.geom_xpos[tip]-origin)@rotation
  d.qpos[21]=.3;d.qpos[23]=.3;d.qpos[24]=.3;mujoco.mj_forward(m,d)
  after=(d.geom_xpos[tip]-origin)@rotation
  rows.append(dict(motion=motion,palm_geom=m.geom(gid).name,palm_local_extent=np.ptp(points,axis=0).tolist(),palm_local_pca_thin_axis=thin.tolist(),palm_thin_eigenvalues=vals.tolist(),open_tip_local=before.tolist(),flexed_tip_local=after.tolist(),positive_flex_tip_displacement_local=(after-before).tolist(),roll_code_capture_axis='link6 local +Y' if motion=='roll' else None,bounce_code_capture_axis='link6 local +Z' if motion=='bounce' else None))
 finally:env.close()
(OUT/'palm_axes.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
