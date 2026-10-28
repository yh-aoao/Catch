import json,sys,itertools
import numpy as np,mujoco
from pathlib import Path
from basket_grip_sweep import make_env,configure,candidates,ThrowActor
actor=ThrowActor('/home/isee/Catch/catch_it_copy/outputs/BasketHold_03_06/2026-10-28/11:38:10/nn/throw_basket_catch_best_reward_-3.42.pth')
env=make_env();c=candidates()[4];rows=[]
m=env.Dcmm.model;body=m.body('link6').id
cols=[int(m.jnt_dofadr[np.flatnonzero(m.jnt_qposadr==i)[0]]) for i in range(15,18)]
try:
 for speed,duration,seed in [(0.,0.,i) for i in range(3)]+list(itertools.product([.4,.8,1.2],[.12,.24,.36],range(3))):
  configure(env,c,1.);np.random.seed(123+seed);obs,_=env.reset(seed=123+seed)
  for _ in range(200):
   obs,_,t,tr,info=env.step(dict(base=np.zeros(2),arm=np.zeros(6),hand=np.zeros(12)))
   if env.basket_mobile_state['unlocked'] or t or tr:break
  start=env.Dcmm.data.body('link6').xpos.copy();maxdrive=0.
  for i in range(90):
   a=actor.action(obs)
   jac=np.zeros((3,m.nv));mujoco.mj_jacBody(m,env.Dcmm.data,jac,None,body)
   direction=env.basket_center-start;direction[2]=0.;direction/=np.linalg.norm(direction)
   maxdrive=max(maxdrive,float((env.Dcmm.data.body('link6').xpos-start)@direction))
   if i*.04<duration:
    desired=speed*direction+np.array([0.,0.,speed*.35])
    qdot=np.linalg.solve(jac[:,cols].T@jac[:,cols]+.02*np.eye(3),jac[:,cols].T@desired)
    a[:3]=np.clip(qdot/np.array([2.,3.,3.]),-1,1);a[6:]=0.
   obs,_,t,tr,info=env.step(dict(base=np.zeros(2),arm=a[:6]*.025,hand=a[6:]*.15))
   if t or tr:break
  r=dict(speed=speed,duration=duration,seed=seed,hit=info['safe_basket_success'],collision=info['robot_dangerous_collision'],distance=env.basket_simple_state.get('best_distance',0.),drive=maxdrive,reason=info.get('terminated_reason'))
  rows.append(r);print(r,flush=True)
  Path('/home/isee/Catch/analysis_artifacts/basket_grip_20261028/arm_probe.json').write_text(json.dumps(rows,indent=2))
finally:env.close()
