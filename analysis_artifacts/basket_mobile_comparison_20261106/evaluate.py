import os,sys,json,time,hashlib
from pathlib import Path
os.chdir('/home/isee/Catch/catch_it_copy');sys.path.insert(0,os.getcwd())
import numpy as np,torch,gymnasium as gym
from hydra import initialize_config_dir,compose
from omegaconf import OmegaConf
from gym_dcmm.envs.basket_mixed import BasketMixedEnv,NAMES
from gym_dcmm.envs.DcmmVecEnv import DcmmVecEnv
from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_basket_mobile_joint import PPO_Basket_MobileJoint
import gym_dcmm.utils.basket_mobile as mobile
torch.set_num_threads(1)
KIND,BALL=sys.argv[1:3]
OUT=Path('/home/isee/Catch/analysis_artifacts/basket_mobile_comparison_20261106')
key=KIND+'_'+BALL
CHECKPOINT=('/home/isee/Catch/11-5/throw_basket_catch_best_reward_37.56.pth' if KIND=='joint' else {
 'original':'/home/isee/Catch/11-4/throw_basket_catch_best_reward_47.35.pth',
 'tennis_ball':'/home/isee/Catch/11-6/throw_basket_catch_best_reward_43.08.pth',
 'baseball':'/home/isee/Catch/11-5/throw_basket_catch_best_reward_49.19.pth'}[BALL])
CONFIG='basket_mobile_three_ball' if KIND=='joint' else {'original':'basket_mobile41','tennis_ball':'basket_mobile_tennis','baseball':'basket_mobile_baseball'}[BALL]
OmegaConf.register_new_resolver('resolve_default',lambda default,arg:default if arg=='' else arg,replace=True)
with initialize_config_dir(config_dir=os.getcwd()+'/configs',version_base='1.1'):
 c=compose(config_name=CONFIG)
c.test=True;c.rl_device='cpu';c.num_envs=1;c.train.ppo.num_actors=1
c.basket_mobile.fixed_throw_probability=0.
keys=['object_motion','basket_fixed_base_training','basket_control_probe','basket_arm_control','basket_reward_version','basket_fixed_physical','basket_carry_reward','basket_carry_options','basket_finger_reward','basket_finger_options','basket_fixed_hand_flex']
if KIND=='joint' or BALL!='original':
 keys+=['basket_delivery_version','basket_delivery_options','basket_mixed_ball','basket_mixed_options']
kwargs={k:OmegaConf.to_container(c[k],resolve=True) if OmegaConf.is_config(c[k]) else c[k] for k in keys}
kwargs.update(task='Catching',render_mode=None,viewer=False,imshow_cam=False,basket_log=False,
 basket_mobile_options=OmegaConf.to_container(c.basket_mobile,resolve=True))
Env=DcmmVecEnv if KIND=='single' and BALL=='original' else BasketMixedEnv
env=gym.vector.SyncVectorEnv([lambda:Env(**kwargs)])
raw=env.envs[0].unwrapped
agent=PPO_Basket_MobileJoint(env,str(OUT/('runtime_'+key)),c)
agent.restore_test(CHECKPOINT);agent.set_eval()
if KIND=='joint':
 # Keep the trained mixed53 interface. Force ball sampling only, without changing physics.
 raw.mixed_options['weights']=[float(n==BALL) for n in NAMES]
def plain(x):
 if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [plain(v) for v in x]
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 return x
records=[];started=time.time()
try:
 for episode in range(200):
  seed=20000+episode;np.random.seed(seed);torch.manual_seed(seed)
  rng=np.random.default_rng(seed)
  target=np.array([rng.uniform(-.2,.2),rng.uniform(1.15,1.55),raw.basket_cfg.basket_center[2]])
  # Ball sampling consumes RNG in the mixed env. Explicit paired targets avoid this confound.
  mobile.sample_target=lambda cfg,rng:target.copy()
  obs,info=raw.reset(seed=seed)
  assert np.allclose(raw.basket_center,target)
  if hasattr(raw,'ball_name'):assert raw.ball_name==BALL
  assert not raw.basket_rehearsal_episode
  total=0.
  for step in range(350):
   batch={k:({a:np.expand_dims(b,0) for a,b in v.items()} if isinstance(v,dict) else np.expand_dims(v,0)) for k,v in obs.items()}
   with torch.no_grad():
    state=agent.obs2tensor(batch);agent.obs={'obs':state}
    action=agent.model.act_inference(dict(obs=agent.running_mean_std(state))).clamp(-1,1)
    command=agent.action2dict(action)
   obs,reward,done,truncated,info=raw.step({k:v[0] for k,v in command.items()})
   total+=reward
   if done or truncated:
    records.append(dict(seed=seed,target=target.tolist(),success=bool(info.get('safe_basket_success')),
      reason=info.get('terminated_reason'),steps=step+1,reward=float(total),
      collision=bool(info.get('robot_dangerous_collision')),control=plain(info.get('basket_control')),
      terms=plain(info.get('basket_reward_totals'))))
    break
  else:raise RuntimeError('No termination at 350 steps')
  (OUT/(key+'.json')).write_text(json.dumps(dict(checkpoint=CHECKPOINT,kind=KIND,ball=BALL,records=records),indent=2))
  if (episode+1)%10==0:
   print(key,len(records),sum(r['success'] for r in records),'elapsed',round(time.time()-started),flush=True)
finally:
 env.close();agent.writer.close()
