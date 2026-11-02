import os,sys
os.chdir('/home/isee/Catch/catch_it_copy');sys.path.insert(0,os.getcwd())
import functools
from unittest.mock import patch
import numpy as np,torch,gymnasium as gym
from hydra import initialize_config_dir,compose
from omegaconf import OmegaConf
from gym_dcmm.envs.basket_mixed import BasketMixedEnv
from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_basket_mixed import PPO_Basket_Mixed
OmegaConf.register_new_resolver('resolve_default',lambda default,arg:default if arg=='' else arg,replace=True)
if __name__=='__main__':
 for config,ball in [('basket_three_ball_ballistic','mixed')]:
  with initialize_config_dir(config_dir=os.getcwd()+'/configs',version_base='1.1'):
   c=compose(config_name=config)
  assert not any([c.checkpoint_catching,c.checkpoint_tracking,c.basket_warm_start,c.basket_mixed_source])
  c.rl_device='cpu';c.num_envs=2;c.train.ppo.num_actors=2;c.train.ppo.minibatch_size=128
  keys=['object_motion','basket_fixed_base_training','basket_control_probe','basket_arm_control','basket_reward_version','basket_fixed_physical','basket_carry_reward','basket_carry_options','basket_finger_reward','basket_finger_options','basket_fixed_hand_flex','basket_delivery_version','basket_delivery_options']
  kwargs={k:OmegaConf.to_container(c[k],resolve=True) if OmegaConf.is_config(c[k]) else c[k] for k in keys}
  kwargs.update(task='Catching',render_mode=None,viewer=False,imshow_cam=False,
    basket_mixed_ball=c.basket_mixed_ball,basket_mixed_options=OmegaConf.to_container(c.basket_mixed_options))
  env=gym.vector.AsyncVectorEnv([functools.partial(BasketMixedEnv,**kwargs)]*2,context='spawn')
  agent=loaded=None
  try:
   with patch('torch.load',side_effect=AssertionError('Single-ball scratch must never load source weights')):
    agent=PPO_Basket_Mixed(env,'/tmp/'+config,c)
   agent.set_eval()
   assert agent.reference is None and not agent.input_normalization_frozen
   assert agent.obs_shape==(37,) and agent.actions_num==18
   agent.save('/tmp/'+config+'_untrained')
   c.test=True
   loaded=PPO_Basket_Mixed(env,'/tmp/'+config+'_reload',c)
   loaded.restore_test('/tmp/'+config+'_untrained.pth');loaded.set_eval()
   obs,_=env.reset(seed=123)
   observed=set()
   for index in range(12):
    if index%4==0:obs,_=env.reset(seed=123+index)
    loaded.obs={'obs':loaded.obs2tensor(obs)}
    expected=np.array(env.call('ball_index'))
    assert np.all((expected>=0)&(expected<3))
    observed.update(expected.tolist())
    loaded.storage.update_data('obses',index,loaded.obs['obs']);loaded.store_rollout_context(index)
    actions=loaded.model_act(loaded.obs,inference=True)['actions'].clamp(-1,1)
    commands=loaded.action2dict(actions)
    np.testing.assert_array_equal(commands['base'],np.zeros((2,2)))
    obs,r,done,trunc,info=env.step(commands);assert np.isfinite(r).all()
   assert len(observed)>=2
   print('Observed mixed ball IDs:',sorted(observed),flush=True)
   print('PASS',config,'scratch/no-source; correct ball; ballistic_v2; 2 spawn workers; 37/18 storage; save/reload; no optimizer updates',flush=True)
  finally:
   env.close()
   if agent:agent.writer.close()
   if loaded:loaded.writer.close()
