import os,sys
os.chdir('/home/isee/Catch/catch_it_copy');sys.path.insert(0,os.getcwd())
from gym_dcmm.envs.basket_mixed import BasketMixedEnv
kwargs=dict(task='Catching',object_motion='throw_basket',basket_fixed_base_training=True,
    basket_fixed_physical=True,basket_carry_reward=True,basket_finger_reward=True,basket_reward_version='basket_aim_v1',
    basket_arm_control='joint_velocity',basket_fixed_hand_flex=[.8,.7,.7],render_mode=None,viewer=False,imshow_cam=False)
env=BasketMixedEnv(**kwargs)
try:
 for ball in ('original','tennis_ball','baseball'):
  env.ball_mode=ball
  for seed in range(20):
   obs,info=env.reset(seed=seed)
   assert env.obs_c_dim==37
   assert len(obs['basket_ball_properties'])==2
  print('READY',ball,20,env.basket_preparation,flush=True)
finally:env.close()
