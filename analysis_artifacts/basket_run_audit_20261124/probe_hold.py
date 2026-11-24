"""Matched reset-state inference ablation. No training or optimizer updates."""
import os,sys,json,hashlib,multiprocessing as mp
from pathlib import Path
ROOT=Path('/home/isee/Catch/catch_it_copy');OUT=Path(__file__).parent
sys.path.insert(0,str(ROOT));os.chdir(ROOT)
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['MPLCONFIGDIR']='/tmp/catch_mpl'

def seeds():
 import numpy as np
 groups={t:[] for t in ('throw','roll','bounce')}
 for seed in range(88000,89000):
  rng=np.random.default_rng(seed);mode=rng.choice(['recorded','near','generic'],p=[.60,.35,.05])
  if mode=='generic':continue
  group=rng.choice(['throw','roll','bounce'])
  if len(groups[group])<4:groups[group].append(seed)
  if all(len(s)==4 for s in groups.values()):break
 return sorted(sum(groups.values(),[]))

def probe(spec):
 import numpy as np,torch,random
 from basket_three_stage.environment import AutonomousTeacherEnv
 from basket_three_stage.policy import Policy
 torch.set_num_threads(1);name,which,temperature=spec
 directory=ROOT/'basket_three_stage/tennis_stage1_hold_seed123'
 ck=torch.load(directory/('teacher_'+('best' if which in ('zero','initial','best') else 'last')+'.pth'),map_location='cpu',weights_only=False)
 cfg=ck['config'];opt=cfg['options'];opt['evaluation']=True
 torch.manual_seed(123);model=Policy()
 if which in ('best','last'):model.load_state_dict(ck['model'],strict=True)
 model.eval();np.random.seed(123);random.seed(123);env=AutonomousTeacherEnv(opt);records=[]
 try:
  for seed in seeds():
   np.random.seed(seed);random.seed(seed);torch.manual_seed(seed)
   obs,initial=env.reset(seed=seed)
   initial_hash=hashlib.sha256(env.Dcmm.data.qpos.tobytes()+env.Dcmm.data.qvel.tobytes()).hexdigest()
   total=0.;stable_max=0.;contact=0;slips=[];actions=[];first_release=None;trace=[]
   for step in range(110):
    with torch.inference_mode():
     if which=='zero':action=np.zeros(20,np.float32)
     elif temperature:
      state=model.normalize(torch.as_tensor(obs)[None]);dist=model.distribution(state)
      action=(dist.loc+temperature*dist.scale*torch.randn_like(dist.loc)).tanh()[0].numpy()
     else:action=model.act(torch.as_tensor(obs)[None])[0].numpy()
    obs,r,done,trunc,info=env.step(action);total+=r
    stable_max=max(stable_max,info['stable_hold_seconds']);contact+=int(info['hand_contact'])
    slips.append(float(np.linalg.norm(env._carry_measurements()[1])));actions.append(float(np.mean(np.abs(action[2:]))))
    if info['released'] and first_release is None:first_release=info['elapsed']
    trace.append(dict(t=info['elapsed'],slip=slips[-1],contact=info['hand_contact'],stable=info['stable_hold_seconds'],released=info['released']))
    if done or trunc:break
   records.append(dict(seed=seed,initial_hash=initial_hash,source=initial['reset_source_task'],mode=initial['reset_mode'],
    success=bool(info['success']),reason=info['terminated_reason'],steps=step+1,reward=total,max_stable=stable_max,
    required_hold=info['required_hold_seconds'],contact_fraction=contact/(step+1),mean_slip=float(np.mean(slips)),
    mean_action_magnitude=float(np.mean(actions)),first_release=first_release,trace=trace))
 finally:env.close()
 (OUT/(name+'.json')).write_text(json.dumps(records,indent=2));print(name,sum(r['success'] for r in records),'/',len(records),flush=True)

if __name__=='__main__':
 specs=[('zero','zero',0),('initial_mean','initial',0),('best_mean','best',0),('best_noise01','best',.1),('best_noise03','best',.3),('best_noise10','best',1),('last_mean','last',0),('last_noise01','last',.1),('last_noise10','last',1)]
 with mp.get_context('spawn').Pool(4) as pool:pool.map(probe,specs)
