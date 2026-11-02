"""Run the three-ball loss path on synthetic rollouts with optimizer.step mocked."""
import copy
import unittest
from unittest.mock import Mock
import torch
import numpy as np
from gym_dcmm.algs.ppo_dcmm.models_track import ActorCritic
from gym_dcmm.algs.ppo_dcmm.basket_mixed_policy import MixedNormalizer
from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_basket_mixed import PPO_Basket_Mixed
from gym_dcmm.algs.ppo_dcmm.experience import ExperienceBuffer
from gym_dcmm.algs.ppo_dcmm.utils import AverageScalarMeter


class MixedUpdateTests(unittest.TestCase):
    def agent(self):
        torch.manual_seed(123)
        a=PPO_Basket_Mixed.__new__(PPO_Basket_Mixed)
        a.model=ActorCritic(dict(actor_units=[16,8],actions_num=18,input_shape=(37,),separate_value_mlp=True))
        a.running_mean_std=MixedNormalizer().eval();a.reference=copy.deepcopy(a.model).requires_grad_(False)
        a.storage=ExperienceBuffer(1,12,12,4,37,18,'cpu')
        raw=torch.randn(12,37);labels=torch.tensor([0]*4+[1]*4+[2]*4)
        raw[:,-2:]=torch.tensor([[.04,.05]]*4+[[.033,.058]]*4+[[.036,.138]]*4)
        with torch.no_grad():old=a.model.act(dict(obs=a.running_mean_std(raw)))
        a.storage.data_dict=dict(obses=raw,actions=old['actions'],mus=old['mus'],sigmas=old['sigmas'],
            neglogpacs=old['neglogpacs'],values=old['values'],returns=old['values']+torch.arange(12.)[:,None],
            advantages=torch.zeros(12),ball_id=labels)
        a.optimizer=torch.optim.Adam(a.model.parameters(),lr=3e-5)
        a.gradients=[]
        def no_update():a.gradients.append(float(a.model.mu.weight.grad.abs().sum()) if a.model.mu.weight.grad is not None else 0.)
        a.optimizer.step=Mock(side_effect=no_update)
        a.set_eval=lambda:(a.model.eval(),a.running_mean_std.eval())
        a.set_train=lambda:a.model.train();a.play_steps=lambda:None
        a.input_normalization_frozen=True
        a.normalize_advantage=True;a.mini_epochs_num=2;a.e_clip=.2;a.clip_value_loss=True
        a.critic_coef=4.;a.entropy_coef=0.;a.bounds_loss_coef=.0001;a.reference_weight=.05
        a.truncate_grads=True;a.grad_norm=1.;a.kl_limit=.01
        a.data_collect_time=a.rl_train_time=0.;a.extra_info={}
        return a

    def test_fixed_statistics_immutable_behavior_and_all_balls_receive_policy_gradients(self):
        a=self.agent()
        stats=copy.deepcopy(a.running_mean_std.state_dict())
        old={k:a.storage.data_dict[k].clone() for k in ('mus','sigmas','neglogpacs')}
        weights=copy.deepcopy(a.model.state_dict())
        a.train_epoch()
        self.assertEqual(len(a.gradients),6)
        self.assertTrue(all(gradient>0 for gradient in a.gradients))
        for k,v in stats.items():torch.testing.assert_close(a.running_mean_std.state_dict()[k],v)
        for k,v in old.items():torch.testing.assert_close(a.storage.data_dict[k],v)
        for k,v in weights.items():torch.testing.assert_close(a.model.state_dict()[k],v)
        self.assertTrue(all(p.grad is None for p in a.reference.parameters()))
        for name in ('original','tennis_ball','baseball'):
            self.assertEqual(a.extra_info['basket_mixed/'+name+'_samples'],4)

    def test_scratch_has_no_teacher_and_learns_fresh_statistics_after_ppo(self):
        a=self.agent();a.reference=None;a.reference_weight=0.;a.input_normalization_frozen=False
        before=float(a.running_mean_std.original.count)
        a.train_epoch()
        self.assertEqual(float(a.running_mean_std.original.count),before+12)
        self.assertTrue(all(gradient>0 for gradient in a.gradients))

    def test_terminal_metrics_use_finished_ball_not_autoreset_next_ball(self):
        a=self.agent();a.per_ball=[AverageScalarMeter(200) for _ in range(3)]
        infos=dict(basket_ball_index=np.array([2]),
            final_info=np.array([dict(basket_ball_index=1,safe_basket_success=True)],dtype=object),
            _final_info=np.array([True]))
        a._update_mobile_metrics(infos,np.array([True]))
        self.assertEqual(len(a.per_ball[1]),1)
        self.assertEqual(a.per_ball[1].get_mean(),1.)
        self.assertEqual(len(a.per_ball[2]),0)

    def test_any_ball_excessive_kl_stops_shared_actor(self):
        a=self.agent();a.storage.data_dict['mus']+=1.
        a.train_epoch()
        self.assertTrue(all(gradient==0 for gradient in a.gradients))
        self.assertEqual(a.extra_info['basket_mixed/actor_stopped'],1.)

if __name__=='__main__':unittest.main()
