from pathlib import Path
import sys,json,hashlib,random
sys.path.insert(0,'/home/isee/Catch/catch_it_copy')
import numpy as np
from gym_dcmm.envs.DcmmVecEnv import DcmmVecEnv
from configs.env import DcmmCfg
OUT=Path(__file__).resolve().parent
np.random.seed(1900);random.seed(1900)
env=DcmmVecEnv(task='Tracking',object_motion='throw',viewer=False,render_mode=None,object_eval=False)
rows=[]
try:
 for seed in range(71000,71010):
  obs,info=env.reset(seed=seed);m=env.Dcmm.model;bid=m.body('object').id;jid=m.body_jntadr[bid];dof=m.jnt_dofadr[jid];gid=env.object_id
  rows.append(dict(seed=seed,mass=float(m.body_mass[bid]),radius=float(m.geom_size[gid,0]),inertia=m.body_inertia[bid].tolist(),joint_damping=m.dof_damping[dof:dof+6].tolist(),initial_position=env.object_pos3d.tolist(),launch_velocity=env.object_vel6d.tolist(),static_seconds=float(env.object_static_time),pid_unit_samples=dict(arm=env.k_arm.tolist(),drive=env.k_drive.tolist(),steer=env.k_steer.tolist(),hand=env.k_hand.tolist())))
finally:env.close()
assert len({r['mass'] for r in rows})==len({r['radius'] for r in rows})==1
assert len({r['joint_damping'][0] for r in rows})==1
assert len({tuple(r['initial_position']) for r in rows})==10
out=dict(active_shape=DcmmCfg.throw_object_shape,configured_mass=DcmmCfg.object_mass.tolist(),configured_radius=DcmmCfg.object_size['sphere'].tolist(),configured_joint_damping=DcmmCfg.object_damping.tolist(),rows=rows)
(OUT/'throw_runtime_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
