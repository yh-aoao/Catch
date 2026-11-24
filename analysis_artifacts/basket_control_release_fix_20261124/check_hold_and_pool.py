import os,sys,json
from pathlib import Path
ROOT=Path('/home/isee/Catch/catch_it_copy');OUT=Path(__file__).parent
os.chdir(ROOT);sys.path.insert(0,str(ROOT))
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ['MPLCONFIGDIR']='/tmp/catch_mpl'

def main():
 import numpy as np,torch
 from basket_autonomous.environment import AutonomousTeacherEnv
 from basket_autonomous.pool import Pool
 ck=torch.load(ROOT/'basket_autonomous/tennis_hold_v2_fixed_seed123_new/teacher_last.pth',map_location='cpu',weights_only=False)
 options=ck['config']['options'];options.update(arm_velocity_protocol='integrated_target_v2',release_protocol='separated_flight_v2')
 np.random.seed(123);env=AutonomousTeacherEnv(options);rows=[]
 try:
  for seed in range(500123,500131):
   np.random.seed(seed);env.reset(seed=seed);initial=env.Dcmm.target_arm_qpos.copy()
   for step in range(50):
    _,r,done,trunc,info=env.step(np.zeros(20,np.float32))
    if done or trunc:break
   rows.append(dict(seed=seed,seconds=info['elapsed'],reason=info['terminated_reason'],contact=info['hand_contact'],release=info['released'],target_drift=float(np.max(np.abs(env.Dcmm.target_arm_qpos-initial)))))
 finally:env.close()
 (OUT/'zero_hold.json').write_text(json.dumps(rows,indent=2));print('2 second zero-command hold:',sum(r['seconds']>1.99 and r['contact'] and r['reason'] is None for r in rows),'/',len(rows),flush=True)
 for stage in ('fixed','mobile'):
  cfg=json.loads((ROOT/'basket_autonomous/configs'/f'{stage}_carry_v2.json').read_text());opt=cfg['options'];opt['episode_seconds']=.04
  if stage=='mobile':opt['fixed_replay_probability']=0.
  pool=Pool(2,opt,123)
  try:
   for _ in range(3):
    obs,r,term,trunc,infos,final=pool.step(np.full((2,20),.1,np.float32))
    assert (term|trunc).all();assert np.isfinite(obs).all();assert np.isfinite(r).all()
    assert all(i['arm_velocity_protocol']=='integrated_target_v2' and i['release_protocol']=='separated_flight_v2' for i in infos)
   print('PASS',stage,'spawn pool/reset with both fixes',flush=True)
  finally:pool.close()
if __name__=='__main__':main()
