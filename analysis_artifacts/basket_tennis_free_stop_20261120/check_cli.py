import os,sys,runpy
if __name__=='__main__':
 os.chdir('/home/isee/Catch/catch_it_copy');sys.path.insert(0,os.getcwd())
 import torch,numpy as np
 from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_basket_mixed import PPO_Basket_Mixed
 from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_basket_mobile_joint import PPO_Basket_MobileJoint
 stage=sys.argv[1]
 def check(self):
  self.set_eval()
  obs,_=self.env.reset(seed=321)
  assert set(self.env.call('ball_name'))=={'tennis_ball'}
  self.obs={'obs':self.obs2tensor(obs)}
  expected_dim=37 if stage=='1' else 51
  assert self.obs['obs'].shape==(2,expected_dim)
  with torch.no_grad():
   action=self.model.act_inference(dict(obs=self.running_mean_std(self.obs['obs']))).clamp(-1,1)
  _,reward,_,_,_=self.env.step(self.action2dict(action))
  assert np.isfinite(reward).all()
  if stage=='1':
   assert self.source_metadata=={'kind':'scratch'}
   self.save('/tmp/basket_tennis_range_untrained_smoke')
  print('PASS stage',stage,'async tennis-only inference; NO training',flush=True)
  self.env.close();self.writer.close()
 PPO_Basket_Mixed.train=check
 PPO_Basket_MobileJoint.train=check
 config='basket_tennis_range' if stage=='1' else 'basket_mobile_tennis_free_stop'
 sys.argv=['train_DCMM.py','--config-name',config,'test=False','num_envs=2','vector_mode=async',
  'viewer=false','basket_log=false','train.ppo.minibatch_size=64','device_id=-1',
  'output_name=BasketTennisWorkflowConstructorCheck']
 if stage=='2':sys.argv+=['basket_throw_checkpoint=/tmp/basket_tennis_range_untrained_smoke.pth']
 runpy.run_path('train_DCMM.py',run_name='__main__')
