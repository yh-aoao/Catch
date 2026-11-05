import os,sys,json
from pathlib import Path
os.chdir('/home/isee/Catch/catch_it_copy');sys.path.insert(0,os.getcwd())
import numpy as np,torch,gymnasium as gym
from hydra import initialize_config_dir,compose
from omegaconf import OmegaConf
from gym_dcmm.envs.DcmmVecEnv import DcmmVecEnv as BasketMixedEnv
from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_basket_mobile_joint import PPO_Basket_MobileJoint as PPO_Basket_Mixed
from gym_dcmm.utils.basket_grip import palm_coordinates
BALL=sys.argv[1];OUT=Path('/home/isee/Catch/analysis_artifacts/basket_original_mobile4735_eval')
CHECKPOINT='/home/isee/Catch/11-4/throw_basket_catch_best_reward_47.35.pth'
OmegaConf.register_new_resolver('resolve_default',lambda default,arg:default if arg=='' else arg,replace=True)
with initialize_config_dir(config_dir=os.getcwd()+'/configs',version_base='1.1'):
 c=compose(config_name='basket_mobile41')
c.test=True;c.rl_device='cpu';c.num_envs=1;c.train.ppo.num_actors=1;c.basket_mobile.fixed_throw_probability=0.0
kwargs={k:OmegaConf.to_container(c[k],resolve=True) if OmegaConf.is_config(c[k]) else c[k] for k in ['object_motion','basket_fixed_base_training','basket_control_probe','basket_arm_control','basket_reward_version','basket_fixed_physical','basket_carry_reward','basket_carry_options','basket_finger_reward','basket_finger_options','basket_fixed_hand_flex']}
kwargs.update(task='Catching',render_mode=None,viewer=False,imshow_cam=False,
    basket_mobile_options=OmegaConf.to_container(c.basket_mobile,resolve=True),basket_log=False)
env=gym.vector.SyncVectorEnv([lambda:BasketMixedEnv(**kwargs)]);raw=env.envs[0].unwrapped
agent=PPO_Basket_Mixed(env,str(OUT/('runtime_'+BALL)),c)
agent.restore_test(CHECKPOINT);agent.set_eval()
records=[]
def plain(x):
 if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [plain(v) for v in x]
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 return x
try:
 for mode in ('mean',):
  for episode in range(100):
   seed=1000+episode;np.random.seed(seed);torch.manual_seed(seed)
   obs,info=raw.reset(seed=seed);trace=[];total=0.
   preparation=plain(raw.basket_preparation)
   for step in range(350):
    batch={k:({a:np.expand_dims(b,0) for a,b in v.items()} if isinstance(v,dict) else np.expand_dims(v,0)) for k,v in obs.items()}
    agent.obs={'obs':agent.obs2tensor(batch)}
    with torch.no_grad():
     processed=agent.running_mean_std(agent.obs['obs'])
     mu=agent.model.act_inference(dict(obs=processed))
     actions=mu if mode=='mean' else agent.model.act(dict(obs=processed))['actions']
     clipped=actions.clamp(-1,1)
     commands=agent.action2dict(clipped)
    obs,reward,done,truncated,info=raw.step({k:v[0] for k,v in commands.items()})
    total+=reward
    trace.append(dict(t=float(raw.Dcmm.data.time-raw.start_time),
       ball=raw.Dcmm.data.qpos[37:40].tolist(),velocity=raw.Dcmm.data.qvel[36:39].tolist(),
       palm=raw.Dcmm.data.body('link6').xpos.tolist(),
       local=palm_coordinates(raw.Dcmm.data,raw.Dcmm.data.qpos[37:40]).tolist(),
       mean=mu[0].cpu().tolist(),action=clipped[0].cpu().tolist(),phase=raw.basket_phase))
    if done or truncated:
     record=dict(ball=BALL,mode=mode,seed=seed,steps=step+1,reward=float(total),
        reason=info.get('terminated_reason'),success=bool(info.get('safe_basket_success')),
        launch=bool(info.get('launch_success')),control=plain(info.get('basket_control')),
        terms=plain(info.get('basket_reward_totals')),candidate=plain(raw.basket_simple_state.get('candidate')),
        last_contact=plain(raw.basket_last_contact),preparation=preparation)
     records.append(record);break
   else:raise RuntimeError('Episode did not terminate within 300 policy steps')
   if (episode+1)%5==0:
    selected=[r for r in records if r['mode']==mode]
    print(BALL,mode,len(selected),'success',sum(r['success'] for r in selected),'launch',sum(r['launch'] for r in selected),flush=True)
 finally_dummy=0
finally:
 env.close();agent.writer.close()
 (OUT/(BALL+'.json')).write_text(json.dumps(records,indent=2))
