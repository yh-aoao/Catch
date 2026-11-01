import os,sys
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
os.chdir('/home/isee/Catch/catch_it_copy');sys.path.insert(0,os.getcwd())
import numpy as np
import torch
import gymnasium as gym
from gym_dcmm.envs.DcmmVecEnv import DcmmVecEnv
from hydra import initialize_config_dir,compose
from omegaconf import OmegaConf
OmegaConf.register_new_resolver('resolve_default',lambda default,arg:default if arg=='' else arg,replace=True)
with initialize_config_dir(config_dir=os.getcwd()+'/configs',version_base='1.1'):
 c=compose(config_name='basket_mobile41')
kwargs={k:OmegaConf.to_container(c[k],resolve=True) if OmegaConf.is_config(c[k]) else c[k] for k in ['object_motion','basket_fixed_base_training','basket_control_probe','basket_arm_control','basket_reward_version','basket_fixed_physical','basket_carry_reward','basket_carry_options','basket_finger_reward','basket_finger_options']}
kwargs.update(task='Catching',render_mode=None,viewer=False,imshow_cam=False,basket_mobile_options=OmegaConf.to_container(c.basket_mobile),basket_log=False)
def make():return DcmmVecEnv(**kwargs)
if __name__=='__main__':
 from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_basket_mobile_joint import PPO_Basket_MobileJoint
 c.rl_device='cpu';c.num_envs=2;c.train.ppo.num_actors=2;c.train.ppo.minibatch_size=128
 env=gym.vector.AsyncVectorEnv([make,make],context='spawn')
 agent=None
 try:
  agent=PPO_Basket_MobileJoint(env,'/tmp/basket_stable_async_init',c);agent.set_eval()
  assert agent.obs_shape==(51,) and agent.storage.obs_dim==51
  obs,_=env.reset(seed=123)
  for n in range(3):
   agent.obs={'obs':agent.obs2tensor(obs)}
   expected=np.array(env.call('basket_rehearsal_episode'),dtype=bool)
   np.testing.assert_array_equal(agent.current_rehearsal.cpu().numpy(),expected)
   agent.storage.update_data('obses',n,agent.obs['obs'])
   agent.store_rollout_context(n)
   np.testing.assert_array_equal(agent.storage.storage_dict['rehearsal'][n].cpu().numpy(),expected)
   commands=agent.action2dict(agent.model_act(agent.obs,inference=True)['actions'])
   obs,reward,done,truncated,info=env.step(commands)
   assert np.isfinite(reward).all()
  print('PASS: two spawn workers; rehearsal flags match policy observations and rollout metadata; inference steps finite; no training')
 finally:
  env.close()
  if agent is not None:agent.writer.close()
