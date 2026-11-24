import os,sys,json,multiprocessing as mp
from pathlib import Path
ROOT=Path('/home/isee/Catch/catch_it_copy');OUT=Path(__file__).parent
sys.path.insert(0,str(ROOT));os.chdir(ROOT)
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['MUJOCO_GL']='egl';os.environ['MPLCONFIGDIR']='/tmp/catch_mpl'

def probe(spec):
 import numpy as np,torch,random
 from basket_autonomous.environment import AutonomousTeacherEnv
 from basket_autonomous.policy import Policy
 torch.set_num_threads(1)
 name,path=spec;ck=torch.load(path,map_location='cpu',weights_only=False)
 np.random.seed(123);random.seed(123);model=Policy();model.load_state_dict(ck['model']);model.eval()
 env=AutonomousTeacherEnv(ck['config']['options']);records=[]
 try:
  for seed in (500123,501132,502141):
   np.random.seed(seed);obs,initial=env.reset(seed=seed);trace=[];total=0.;gaps=0;recontacts=0;previous_release=False
   for step in range(300):
    with torch.inference_mode():act=model.act(torch.as_tensor(obs)[None])[0].numpy()
    obs,r,done,trunc,info=env.step(act);total+=r
    if env._released_once and not previous_release:gaps+=1
    previous_release=env._released_once
    if env._released_once and info['hand_contact']:recontacts+=1
    trace.append(dict(t=info['elapsed'],touch=info['hand_contact'],release_latched=env._released_once,active_release=env._release is not None,
     ball=env.Dcmm.data.qpos[37:40].tolist(),local=env._carry_measurements()[0].tolist(),base_speed=info['base_speed'],low_speed=info['low_speed_seconds'],
     purposeful=info['purposeful_launch'],action=act.tolist(),terms=info['reward_terms'],slip=env._carry_measurements()[1].tolist()))
    if done or trunc:break
   records.append(dict(seed=seed,initial=initial,steps=step+1,reward=total,final=info,post_release_contact_steps=recontacts,first_gap=next((t for t in trace if t['release_latched']),None),trace=trace))
   print(name,seed,step+1,info['terminated_reason'],'postrelease-contact',recontacts,flush=True)
 finally:env.close()
 (OUT/(name+'.json')).write_text(json.dumps(records,indent=2))

if __name__=='__main__':
 specs=[]
 for name,folder in [('old','tennis_random_hold_fixed_seed123'),('v2','tennis_hold_v2_fixed_seed123_new')]:
  for ck in ('best','last'):specs.append((name+'_'+ck,str(ROOT/'basket_autonomous'/folder/('teacher_'+ck+'.pth'))))
 with mp.get_context('spawn').Pool(4) as pool:pool.map(probe,specs)
