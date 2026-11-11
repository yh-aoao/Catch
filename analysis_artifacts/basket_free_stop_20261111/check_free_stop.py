import os,sys,tempfile
os.chdir('/home/isee/Catch/catch_it_copy');sys.path.insert(0,os.getcwd())
import torch,numpy as np,gymnasium as gym
from hydra import initialize_config_dir,compose
from omegaconf import OmegaConf
from gym_dcmm.envs.basket_free_stop import BasketFreeStopEnv as BasketMixedEnv
from gym_dcmm.envs.basket_mixed import make_options
from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_basket_mobile_joint import PPO_Basket_MobileJoint
from gym_dcmm.algs.ppo_dcmm.models_track import ActorCritic
from gym_dcmm.algs.ppo_dcmm.basket_mixed_policy import MixedNormalizer
OmegaConf.register_new_resolver('resolve_default',lambda default,arg:default if arg=='' else arg,replace=True)
for ball in ('free_stop',):
 with initialize_config_dir(config_dir=os.getcwd()+'/configs',version_base='1.1'):
  c=compose(config_name='basket_mobile_'+ball)
 c.test=False;c.rl_device='cpu';c.num_envs=1;c.train.ppo.num_actors=1;c.train.ppo.minibatch_size=64
 keys=['object_motion','basket_fixed_base_training','basket_control_probe','basket_arm_control','basket_reward_version','basket_fixed_physical','basket_carry_reward','basket_carry_options','basket_finger_reward','basket_finger_options','basket_fixed_hand_flex','basket_delivery_version','basket_delivery_options','basket_mixed_ball','basket_mixed_options','basket_free_stop']
 kwargs={k:OmegaConf.to_container(c[k],resolve=True) if OmegaConf.is_config(c[k]) else c[k] for k in keys}
 kwargs.update(task='Catching',render_mode=None,viewer=False,imshow_cam=False,basket_log=False,basket_mobile_options=OmegaConf.to_container(c.basket_mobile,resolve=True))
 env=gym.vector.SyncVectorEnv([lambda:BasketMixedEnv(**kwargs)])
 raw=env.envs[0].unwrapped
 agent=PPO_Basket_MobileJoint(env,tempfile.mkdtemp(),c)
 agent.set_eval()
 ck=torch.load(c.basket_throw_checkpoint,map_location='cpu',weights_only=False)
 source=ActorCritic(dict(actor_units=[256,128],actions_num=18,input_shape=(37,),separate_value_mlp=True))
 source.load_state_dict(ck['model']);source.eval()
 norm=MixedNormalizer();norm.load_state_dict(ck['running_mean_std']);norm.eval()
 x=torch.randn(20,53);x[:,47]=1
 f=x[:,:35].clone();f[:,34]+=.2
 props=torch.tensor([[.0332757853,.058],[.0361762338,.138],[.04,.05]]).repeat(7,1)[:20]
 x[:,51:53]=props
 expected=source.act_inference(dict(obs=norm(torch.cat((f,props),dim=1))))
 actual=agent.model.act_inference(dict(obs=agent.running_mean_std(x)))[:,2:]
 torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
 assert all(p.requires_grad for p in agent.model.throw_actor.parameters())
 assert raw.obs_c_dim==53
 for seed,probability in ((123,1.),(127,0.)):
  raw.basket_cfg.mobile_fixed_throw_probability=probability
  obs,_=env.reset(seed=seed)
  assert raw.ball_name in ('original','tennis_ball','baseball')
  for _ in range(300):
   agent.obs={'obs':agent.obs2tensor(obs)}
   with torch.no_grad():
    action=agent.model.act_inference(dict(obs=agent.running_mean_std(agent.obs['obs']))).clamp(-1,1)
   obs,r,d,t,info=env.step(agent.action2dict(action))
   assert np.isfinite(r).all()
   last=info['final_info'][0] if 'final_info' in info else None
   if last is not None:
    assert abs(float(r[0])-sum(last['basket_reward_terms'].values()))<1e-4
    assert last['basket_reward_terms']['task_failure']==0.
   if bool(d[0]) or bool(t[0]):break
  else:raise AssertionError('No episode termination')
 seen=set()
 for seed in range(12):
  env.reset(seed=seed)
  seen.add(raw.ball_name)
 assert seen=={'original','tennis_ball','baseball'}
 agent.save('/tmp/'+ball+'_mobile_smoke')
 c.test=True
 loaded=PPO_Basket_MobileJoint(env,tempfile.mkdtemp(),c)
 loaded.restore_test('/tmp/'+ball+'_mobile_smoke.pth')
 print('PASS',ball,'source actions identical; mobile53; trainable arm/hand; inference; save/reload; NO optimizer updates',flush=True)
 agent.writer.close();loaded.writer.close();env.close()
