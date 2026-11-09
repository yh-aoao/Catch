import os,sys,tempfile
os.chdir('/home/isee/Catch/catch_it_copy');sys.path.insert(0,os.getcwd())
import numpy as np,torch,gymnasium as gym
from hydra import initialize_config_dir,compose
from omegaconf import OmegaConf
from gym_dcmm.envs.basket_mixed import BasketMixedEnv
from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_basket_mixed import PPO_Basket_Mixed
OmegaConf.register_new_resolver('resolve_default',lambda default,arg:default if arg=='' else arg,replace=True)
with initialize_config_dir(config_dir=os.getcwd()+'/configs',version_base='1.1'):
 c=compose(config_name='basket_three_ball_range')
c.rl_device='cpu';c.num_envs=1;c.train.ppo.num_actors=1;c.train.ppo.minibatch_size=64
keys=['object_motion','basket_fixed_base_training','basket_control_probe','basket_arm_control','basket_reward_version','basket_fixed_physical','basket_carry_reward','basket_carry_options','basket_finger_reward','basket_finger_options','basket_fixed_hand_flex','basket_delivery_version','basket_delivery_options','basket_mixed_ball','basket_mixed_options','basket_range']
kwargs={k:OmegaConf.to_container(c[k],resolve=True) if OmegaConf.is_config(c[k]) else c[k] for k in keys}
kwargs.update(task='Catching',render_mode=None,viewer=False,imshow_cam=False,basket_log=False)
env=gym.vector.SyncVectorEnv([lambda:BasketMixedEnv(**kwargs)]);raw=env.envs[0].unwrapped
from unittest.mock import patch
with patch('torch.load',side_effect=AssertionError('Scratch constructor must not load weights')):
 a=PPO_Basket_Mixed(env,tempfile.mkdtemp(),c)
a.set_eval()
assert a.source_metadata=={'kind':'scratch'}
assert not a.input_normalization_frozen
assert a.reference is None and all(p.requires_grad for p in a.model.parameters())
assert float(a.running_mean_std.original.count)==1.
assert a.init_lr==.0003
baseline=raw.basket_cfg.basket_center.copy();ys=[];replay=[];balls=set()
for seed in range(20):
 obs,info=env.reset(seed=seed);ys.append(float(raw.basket_center[1]));replay.append(raw.range_rehearsal);balls.add(raw.ball_name)
 assert abs(ys[-1]-baseline[1])<=.100001
 np.testing.assert_array_equal(raw.basket_cfg.basket_center,baseline)
 assert np.allclose(raw.Dcmm.model.body('basket_target').pos,raw.basket_center)
 assert not np.any(raw.Dcmm.data.qfrc_applied)
 with torch.no_grad():
  a.obs={'obs':a.obs2tensor(obs)}
  action=a.model.act_inference(dict(obs=a.running_mean_std(a.obs['obs']))).clamp(-1,1)
 commands=a.action2dict(action)
 np.testing.assert_array_equal(commands['base'],0)
 _,r,_,_,_=env.step(commands);assert np.isfinite(r).all()
assert len(set(ys))>5 and any(replay) and not all(replay) and len(balls)==3
a.save('/tmp/range_smoke')
c.test=True
b=PPO_Basket_Mixed(env,tempfile.mkdtemp(),c);b.restore_test('/tmp/range_smoke.pth')
print('PASS: scratch constructor forbids weight loading, all3 balls, target randomized and static in step, baseline restored, zero base command, finite inference, save/load; no optimizer updates',flush=True)
env.close();a.writer.close();b.writer.close()
