import os,sys
os.chdir('/home/isee/Catch/catch_it_copy');sys.path.insert(0,os.getcwd())
import numpy as np,torch,gymnasium as gym
from hydra import initialize_config_dir,compose
from omegaconf import OmegaConf
from gym_dcmm.envs.basket_mixed import BasketMixedEnv
from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_basket_mixed import PPO_Basket_Mixed
OmegaConf.register_new_resolver('resolve_default',lambda default,arg:default if arg=='' else arg,replace=True)
with initialize_config_dir(config_dir=os.getcwd()+'/configs',version_base='1.1'):
 c=compose(config_name='basket_three_ball')
kwargs={k:OmegaConf.to_container(c[k],resolve=True) if OmegaConf.is_config(c[k]) else c[k] for k in ['object_motion','basket_fixed_base_training','basket_control_probe','basket_arm_control','basket_reward_version','basket_fixed_physical','basket_carry_reward','basket_carry_options','basket_finger_reward','basket_finger_options','basket_fixed_hand_flex']}
kwargs.update(task='Catching',render_mode=None,viewer=False,imshow_cam=False,
    basket_mixed_ball=c.basket_mixed_ball,basket_mixed_options=OmegaConf.to_container(c.basket_mixed_options))
def make():return BasketMixedEnv(**kwargs)
if __name__=='__main__':
 c.rl_device='cpu';c.num_envs=2;c.train.ppo.num_actors=2;c.train.ppo.minibatch_size=128
 env=gym.vector.AsyncVectorEnv([make,make],context='spawn');agent=loaded=None
 try:
  agent=PPO_Basket_Mixed(env,'/tmp/basket_mixed_agent',c);agent.set_eval()
  assert agent.obs_shape==(37,) and agent.actions_num==18 and agent.storage.obs_dim==37
  c.test=True;c.basket_mixed_source='/missing/original_source.pth'
  agent.save('/tmp/basket_mixed_untrained_smoke')
  loaded=PPO_Basket_Mixed(env,'/tmp/basket_mixed_loaded',c)
  loaded.restore_test('/tmp/basket_mixed_untrained_smoke.pth');loaded.set_eval()
  obs,_=env.reset(seed=123)
  for n in range(3):
   loaded.obs={'obs':loaded.obs2tensor(obs)}
   expected=np.array(env.call('ball_index'),dtype=int)
   np.testing.assert_array_equal(loaded.current_ball.cpu().numpy(),expected)
   loaded.store_rollout_context(n);loaded.storage.update_data('obses',n,loaded.obs['obs'])
   np.testing.assert_array_equal(loaded.storage.storage_dict['ball_id'][n].cpu().numpy(),expected)
   commands=loaded.action2dict(loaded.model_act(loaded.obs,inference=True)['actions'])
   np.testing.assert_array_equal(commands['base'],np.zeros((2,2)))
   obs,reward,done,truncated,info=env.step(commands);assert np.isfinite(reward).all()
  print('PASS: 2 spawn workers; source transfer;37 observations/18 actions;fixed base; per-ball metadata;embedded save/reload;finite inference. NO optimizer updates')
 finally:
  env.close()
  if agent:agent.writer.close()
  if loaded:loaded.writer.close()
