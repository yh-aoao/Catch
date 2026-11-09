import os,sys,runpy
if __name__=='__main__':
 os.chdir('/home/isee/Catch/catch_it_copy');sys.path.insert(0,os.getcwd())
 import torch,numpy as np
 from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_basket_mixed import PPO_Basket_Mixed as PPO_Basket_MobileJoint
 def check(self):
  self.set_eval()
  obs,_=self.env.reset(seed=321)
  self.obs={'obs':self.obs2tensor(obs)}
  assert self.obs['obs'].shape==(2,37)
  with torch.no_grad():
   action=self.model.act_inference(dict(obs=self.running_mean_std(self.obs['obs']))).clamp(-1,1)
  _,reward,_,_,_=self.env.step(self.action2dict(action))
  assert np.isfinite(reward).all()
  print('PASS CLI async workers, source initialized, inference only; NO training',flush=True)
  self.env.close();self.writer.close()
 PPO_Basket_MobileJoint.train=check
 sys.argv=['train_DCMM.py','--config-name','basket_three_ball_range','test=False','num_envs=2','vector_mode=async','viewer=false','basket_log=false','train.ppo.minibatch_size=64','device_id=-1','output_name=BasketSingleMobileConstructorCheck']
 runpy.run_path('train_DCMM.py',run_name='__main__')
