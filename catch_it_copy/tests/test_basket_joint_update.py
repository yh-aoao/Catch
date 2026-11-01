"""Exercise the real joint updater on synthetic data; optimizer steps are mocked."""
import copy
import unittest
from unittest.mock import Mock

import torch

from gym_dcmm.algs.ppo_dcmm.basket_mobile_joint import JointActorCritic,JointNormalizer
from gym_dcmm.algs.ppo_dcmm.basket_joint_stability import options
from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_basket_mobile_joint import PPO_Basket_MobileJoint
from gym_dcmm.algs.ppo_dcmm.experience import ExperienceBuffer


class JointUpdateTests(unittest.TestCase):
    def agent(self):
        torch.manual_seed(123)
        a=PPO_Basket_MobileJoint.__new__(PPO_Basket_MobileJoint)
        a.model=JointActorCritic([16,8]);a.running_mean_std=JointNormalizer().eval()
        a.reference_actor=copy.deepcopy(a.model.throw_actor).requires_grad_(False)
        a.reference_mu=copy.deepcopy(a.model.throw_mu).requires_grad_(False)
        a.reference_sigma=a.model.throw_sigma.detach().clone()
        a.joint_options=options();a.extra_info={};a.data_collect_time=a.rl_train_time=0.
        a.normalize_advantage=True;a.mini_epochs_num=2;a.e_clip=.2
        a.clip_value_loss=True;a.critic_coef=4.;a.entropy_coef=0.;a.bounds_loss_coef=.0001
        a.truncate_grads=True;a.grad_norm=1.;a.lr_schedule='fixed'
        a.storage=ExperienceBuffer(1,12,12,4,51,20,'cpu')
        raw=torch.randn(12,51);raw[:,47]=torch.tensor([0.]*4+[1.]*8)
        with torch.no_grad():old=a.model.act(dict(obs=a.running_mean_std(raw)))
        a.storage.data_dict=dict(obses=raw,actions=old['actions'],mus=old['mus'],sigmas=old['sigmas'],
            neglogpacs=old['neglogpacs'],values=old['values'],returns=old['values']+torch.arange(12.)[:,None],
            advantages=torch.zeros(12),rehearsal=torch.tensor([False]*8+[True]*4))
        groups=a.model.actor_groups()
        a.optimizer=torch.optim.Adam([dict(params=g,lr=lr,name=name) for g,lr,name in zip(
            groups,[1e-4,1e-5,1e-4],['base','throw','critic'])])
        a.initial_group_lrs={g['name']:g['lr'] for g in a.optimizer.param_groups}
        a.snapshots=[]
        def no_update():
            a.snapshots.append(dict(count=float(a.running_mean_std.navigation.count),
                gradients=[sum(float(p.grad.abs().sum()) for p in group if p.grad is not None)
                           for group in a.model.actor_groups()]))
        a.optimizer.step=Mock(side_effect=no_update)
        a.set_eval=lambda:(a.model.eval(),a.running_mean_std.eval())
        a.set_train=lambda:a.model.train()
        a.play_steps=lambda:None
        return a

    def test_real_loss_path_preserves_behavior_anchor_and_normalization_and_updates_both_branches(self):
        a=self.agent();data=a.storage.data_dict
        anchor={k:data[k].clone() for k in ('mus','sigmas','neglogpacs')}
        parameters={k:v.clone() for k,v in a.model.state_dict().items()}
        result=a.train_epoch()
        self.assertEqual(len(result[0]),6)
        for key,value in anchor.items():torch.testing.assert_close(data[key],value)
        self.assertTrue(all(s['count']==1. for s in a.snapshots))
        self.assertEqual(float(a.running_mean_std.navigation.count),13.)
        self.assertEqual(float(a.running_mean_std.throwing.count),1.)
        self.assertTrue(any(s['gradients'][0]>0 for s in a.snapshots))
        self.assertTrue(any(s['gradients'][1]>0 for s in a.snapshots))
        self.assertTrue(all(p.grad is None for p in a.reference_actor.parameters()))
        for k,v in parameters.items():torch.testing.assert_close(a.model.state_dict()[k],v)
        self.assertEqual(a.extra_info['basket_joint/replay_throw_samples'],4)

    def test_large_base_kl_stops_only_base_updates(self):
        a=self.agent();a.storage.data_dict['mus'][:4,:2]+=1.
        a.train_epoch()
        self.assertEqual(a.extra_info['basket_joint/base_update_stopped'],1.)
        self.assertTrue(all(s['gradients'][0]==0. for s in a.snapshots))
        self.assertTrue(any(s['gradients'][1]>0. for s in a.snapshots))
        self.assertEqual(a.extra_info['basket_joint/throw_update_stopped'],0.)

    def test_optional_kl_schedule_cannot_raise_learning_rates_above_initial_caps(self):
        a=self.agent();a.lr_schedule='kl'
        for group in a.optimizer.param_groups:group['lr']*=20.
        a.train_epoch()
        for group in a.optimizer.param_groups:
            self.assertLessEqual(group['lr'],a.initial_group_lrs[group['name']])

if __name__=='__main__':unittest.main()
