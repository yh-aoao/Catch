import sys,json,numpy as np,torch,gymnasium as gym
from pathlib import Path
from omegaconf import OmegaConf
import gym_dcmm
from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_basket_two_stage import PPO_Basket_TwoStage
root=Path('/home/isee/Catch/catch_it_copy/outputs/BasketThrowFirst/2026-10-28/01:17:01')
rows={}
for name in ['throw_basket_catch_best_reward_31.80.pth','throw_basket_catch_best_success.pth','throw_basket_catch_last.pth']:
 cfg=OmegaConf.load(root/'resolved_config.yaml');cfg.test=True;cfg.viewer=False;cfg.num_envs=1;cfg.train.ppo.num_actors=1;cfg.rl_device='cpu';cfg.basket_mobile.fixed_throw_probability=0.
 np.random.seed(123);torch.manual_seed(123)
 e=gym.make_vec('gym_dcmm/DcmmVecWorld-v0',num_envs=1,vectorization_mode='sync',task='Catching',object_motion='throw_basket',basket_arm_control='joint_velocity',basket_reward_version='basket_mobile_v1',basket_mobile_options=OmegaConf.to_container(cfg.basket_mobile),viewer=False)
 p=PPO_Basket_TwoStage(e,'/tmp/basket-1028-compare/'+name,cfg);p.restore_test(str(root/'nn'/name));p.set_eval();result=[]
 for seed in range(20):
  np.random.seed(123+seed);o,_=e.reset(seed=123+seed);p.obs={'obs':p.obs2tensor(o)};reward=0.
  for j in range(270):
   with torch.no_grad():a=p.model_act(p.obs,inference=True)['actions'].clamp(-1,1)
   a=torch.nn.functional.pad(a,(0,20-a.shape[1]));o,r,t,tr,i=e.step(p.action2dict(a));reward+=float(r[0]);p.obs={'obs':p.obs2tensor(o)}
   if t[0] or tr[0]:
    f=i['final_info'][0];result.append(dict(seed=seed,success=f['safe_basket_success'],reason=f['terminated_reason'],reward=reward,control=f['basket_control']));break
  print('COMPARE',name,seed,result[-1]['success'],result[-1]['reason'],flush=True)
 rows[name]=result;e.close();p.writer.close()
 print('TOTAL',name,sum(r['success'] for r in result),'/',len(result),flush=True)
 Path('/tmp/basket_1028_compare.json').write_text(json.dumps(rows,indent=2))
