exec(open('/tmp/check_mobile41.py').read().split('env=DcmmVecEnv(**kwargs)')[0])
from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_basket_mobile_joint import PPO_Basket_MobileJoint
c.rl_device='cpu';c.num_envs=1;c.train.ppo.num_actors=1;c.train.ppo.minibatch_size=64
vector=gym.vector.SyncVectorEnv([lambda:DcmmVecEnv(**kwargs)])
try:
 agent=PPO_Basket_MobileJoint(vector,'/tmp/basket_joint_init',c)
 assert agent.actions_num==20 and agent.storage.act_dim==20
 assert agent.optimizer.param_groups[1]['lr']==agent.optimizer.param_groups[0]['lr']*.1
 source=torch.load(c.basket_throw_checkpoint,map_location='cpu',weights_only=False)
 for key,value in source['model'].items():
  if key.startswith('actor_mlp.'):
   torch.testing.assert_close(agent.model.throw_actor.state_dict()[key[len('actor_mlp.'):]],value)
 assert all(p.requires_grad for g in agent.optimizer.param_groups for p in g['params'])
 expected={id(p) for p in agent.model.parameters()}
 registered=[id(p) for g in agent.optimizer.param_groups for p in g['params']]
 assert len(registered)==len(set(registered)) and expected==set(registered)
 agent.save('/tmp/basket_joint_untrained_smoke')
 c.test=True;c.basket_throw_checkpoint='/nonexistent/original_source_not_needed_for_test.pth'
 loaded=PPO_Basket_MobileJoint(vector,'/tmp/basket_joint_reload',c)
 loaded.restore_test('/tmp/basket_joint_untrained_smoke.pth');loaded.set_eval()
 raw,_=vector.reset(seed=123);loaded.obs={'obs':loaded.obs2tensor(raw)}
 result=loaded.model_act(loaded.obs,inference=True)
 commands=loaded.action2dict(result['actions'])
 raw,r,done,trunc,info=vector.step(commands)
 assert np.isfinite(r).all()
 # Only synthetic backward checks; never call optimizer.step or agent.train.
 agent.model.zero_grad(set_to_none=True);agent.running_mean_std.eval()
 mixed=loaded.obs['obs'].repeat(4,1);mixed[:,47]=torch.tensor([0.,1.,0.,1.])
 processed=agent.running_mean_std(mixed);sample=agent.model.act(dict(obs=processed))
 update=agent.model(dict(obs=processed,prev_actions=sample['actions']))
 loss=-(torch.exp(sample['neglogpacs']-update['prev_neglogp'])*torch.tensor([1.,-.4,.7,-.2])).mean()
 loss.backward()
 assert all(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in group) for group in agent.model.actor_groups()[:2])
 print('PASS: source actor equality, all actors trainable/in optimizer, 10:1 LR groups, 20-action storage, mixed-phase PPO gradients, embedded save/reload, vector inference; NO optimizer update')
finally:
 vector.close()
 if 'agent' in locals():agent.writer.close()
 if 'loaded' in locals():loaded.writer.close()
