import copy
import unittest

import torch

from gym_dcmm.algs.ppo_dcmm.models_track import ActorCritic
from gym_dcmm.algs.ppo_dcmm.utils import RunningMeanStd
from gym_dcmm.algs.ppo_dcmm.basket_mobile_joint import JointActorCritic,JointNormalizer
from gym_dcmm.utils.basket_fingers import make_options as finger_options
from gym_dcmm.utils.basket_carry import make_options as carry_options


class JointMobileTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.units=[16,8]
        self.source=ActorCritic(dict(actor_units=self.units,actions_num=18,input_shape=(35,),separate_value_mlp=True))
        self.source_norm=RunningMeanStd((35,)).eval()
        self.source_norm.running_mean.copy_(torch.randn(35)*.2)
        self.source_norm.running_var.copy_(torch.rand(35)+.5)
        self.checkpoint=dict(model=self.source.state_dict(),running_mean_std=self.source_norm.state_dict(),
            basket_fixed_protocol='aim_carry_fingers_v1',basket_arm_control='joint_velocity',
            action_layout='arm6_hand12',basket_finger_options=finger_options(),basket_carry_options=carry_options())
        self.model=JointActorCritic(self.units)
        self.norm=JointNormalizer().eval()
        self.model.import_throw(self.checkpoint,self.norm,finger_options(),carry_options())

    def observations(self,phase):
        raw=torch.randn(len(phase),51);raw[:,47]=torch.tensor(phase,dtype=raw.dtype)
        return raw

    def test_initial_throw_mean_and_std_exactly_match_4130(self):
        raw=self.observations([1,1,1,1]);fixed=raw[:,:35].clone();fixed[:,34]+=.2
        expected=self.source.act_inference(dict(obs=self.source_norm(fixed)))
        actual=self.model.act_inference(dict(obs=self.norm(raw)))
        torch.testing.assert_close(actual[:,2:],expected)
        torch.testing.assert_close(actual[:,:2],torch.zeros(4,2))
        torch.testing.assert_close(self.model.throw_sigma,self.source.sigma)
        self.assertTrue(all(p.requires_grad for p in self.model.throw_actor.parameters()))

    def test_navigation_and_throw_likelihoods_exclude_unexecuted_actions(self):
        obs=self.norm(self.observations([0,1,0,1]))
        sampled=self.model.act(dict(obs=obs))
        mask=self.model.action_mask(obs)
        self.assertTrue(torch.equal(sampled['actions'][~mask],torch.zeros_like(sampled['actions'][~mask])))
        first=self.model(dict(obs=obs,prev_actions=sampled['actions']))
        changed=sampled['actions'].clone();changed[~mask]=100.
        second=self.model(dict(obs=obs,prev_actions=changed))
        torch.testing.assert_close(first['prev_neglogp'],second['prev_neglogp'])
        torch.testing.assert_close(first['prev_neglogp'],sampled['neglogpacs'])
        torch.testing.assert_close(torch.exp(sampled['neglogpacs']-first['prev_neglogp']),torch.ones(4))

    def test_both_actors_receive_nonzero_ppo_gradients_on_mixed_phases(self):
        obs=self.norm(self.observations([0,1,0,1,0,1,0,1]))
        old=self.model.act(dict(obs=obs))
        new=self.model(dict(obs=obs,prev_actions=old['actions']))
        advantages=torch.tensor([1.,-.2,.6,-.8,.3,-1.,.8,.4])
        loss=-(torch.exp(old['neglogpacs']-new['prev_neglogp'])*advantages).mean()
        loss.backward();self.model.clear_inactive_gradients(obs)
        base,throw,_=self.model.actor_groups()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in base if p.grad is not None),0.)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in throw if p.grad is not None),0.)

    def test_single_phase_clears_inactive_actor_gradients_to_prevent_adam_momentum_updates(self):
        for phase,inactive in [(0,1),(1,0)]:
            self.model.zero_grad(set_to_none=True)
            obs=self.norm(self.observations([phase]*4));old=self.model.act(dict(obs=obs))
            self.model(dict(obs=obs,prev_actions=old['actions']))['prev_neglogp'].mean().backward()
            self.model.clear_inactive_gradients(obs)
            self.assertTrue(all(p.grad is None for p in self.model.actor_groups()[inactive]))

    def test_normalizer_updates_only_new_statistics_without_freezing_actor(self):
        snapshot=copy.deepcopy(self.norm.throwing.state_dict());base_count=self.norm.navigation.count.clone()
        self.norm.train();self.norm(self.observations([0,1,0,1]))
        self.assertGreater(self.norm.navigation.count,base_count)
        self.assertFalse(self.norm.throwing.training)
        for k,v in snapshot.items():torch.testing.assert_close(v,self.norm.throwing.state_dict()[k])
        self.assertTrue(self.model.throw_sigma.requires_grad)

    def test_checkpoint_roundtrip_contains_both_trainable_actors_and_normalizers(self):
        raw=self.observations([0,1,0,1]);expected=self.model.act_inference(dict(obs=self.norm(raw)))
        model=JointActorCritic(self.units);normalizer=JointNormalizer().eval()
        model.load_state_dict(self.model.state_dict(),strict=True)
        normalizer.load_state_dict(self.norm.state_dict(),strict=True)
        torch.testing.assert_close(model.act_inference(dict(obs=normalizer(raw))),expected)

    def test_kl_sums_active_dimensions_and_ignores_inactive_channels(self):
        obs=self.norm(self.observations([0]*4));old=self.model.act(dict(obs=obs))
        mu=old['mus'].clone();mu[:,:2]+=.1
        sigma=old['sigmas']
        expected=(.1**2/(2*sigma[:,:2]**2)).sum(-1).mean()
        self.assertAlmostEqual(float(self.model.policy_kl(obs,old['mus'],sigma,old['mus'],sigma)),0.,places=6)
        torch.testing.assert_close(self.model.policy_kl(obs,mu,sigma,old['mus'],sigma),expected)

if __name__=='__main__':unittest.main()
