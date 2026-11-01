exec(open('/tmp/check_mobile41.py').read().split('env=DcmmVecEnv(**kwargs)')[0])
from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_basket_two_stage import PPO_Basket_TwoStage
c.rl_device='cpu';c.num_envs=1;c.train.ppo.num_actors=1;c.train.ppo.minibatch_size=64
vector=gym.vector.SyncVectorEnv([lambda:DcmmVecEnv(**kwargs)])
try:
 agent=PPO_Basket_TwoStage(vector,'/tmp/basket_mobile41_init',c)
 assert agent.actions_num==2 and not agent.input_normalization_frozen
 assert all(not p.requires_grad for p in agent.throw_policy.model.parameters())
 frozen_ids={id(p) for p in agent.throw_policy.model.parameters()}
 assert all(id(p) not in frozen_ids for g in agent.optimizer.param_groups for p in g['params'])
 agent.save('/tmp/mobile41_untrained_smoke')
 frozen_before={k:v.clone() for k,v in agent.throw_policy.model.state_dict().items()}
 c.test=True;c.basket_throw_checkpoint='/nonexistent/test_must_use_embedded_actor.pth'
 loaded=PPO_Basket_TwoStage(vector,'/tmp/basket_mobile41_reload',c)
 loaded.restore_test('/tmp/mobile41_untrained_smoke.pth')
 for k,v in frozen_before.items():torch.testing.assert_close(loaded.throw_policy.model.state_dict()[k],v)
 raw,_=vector.reset(seed=123);loaded.obs={'obs':loaded.obs2tensor(raw)}
 commands=loaded.action2dict(torch.zeros(1,20))
 raw,r,done,trunc,info=vector.step(commands)
 assert np.isfinite(r).all()
 print('PASS: fresh base2 constructor, frozen actor excluded from optimizer, embedded save/reload, vector action step; no optimizer updates')
finally:
 vector.close()
 if 'agent' in locals():agent.writer.close()
 if 'loaded' in locals():loaded.writer.close()
