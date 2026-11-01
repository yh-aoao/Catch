import unittest
import torch
from gym_dcmm.algs.ppo_dcmm.basket_joint_stability import (
    gaussian_kl,balanced_loss,normalize_advantages,bounded_lr,masks,options)


class StabilityTests(unittest.TestCase):
    def test_kl_sums_all_18_actions_and_is_zero_for_identical_policies(self):
        old=torch.zeros(4,18);std=torch.ones_like(old)
        torch.testing.assert_close(gaussian_kl(old,std,old,std),torch.zeros(4))
        torch.testing.assert_close(gaussian_kl(old,std,old+.2,std),torch.full((4,),18*.2**2/2))

    def test_replay_loss_mass_not_diluted_by_many_navigation_or_mobile_samples(self):
        groups=(torch.tensor([True]*100+[False]*11),
                torch.tensor([False]*100+[True]*10+[False]),
                torch.tensor([False]*110+[True]))
        loss=torch.tensor([2.]*100+[4.]*10+[8.],requires_grad=True)
        result=balanced_loss(loss,groups,.5)
        self.assertAlmostEqual(float(result.detach()),2.+.5*4.+.5*8.)
        result.backward()
        self.assertAlmostEqual(float(loss.grad[-1]),.5)
        self.assertAlmostEqual(float(loss.grad[100:110].sum()),.5,places=6)

    def test_rollout_loss_weight_is_invariant_to_replay_clustering_across_minibatches(self):
        value=torch.tensor([2.]*4+[4.]*3+[8.],requires_grad=True)
        groups=(torch.tensor([1,1,1,1,0,0,0,0],dtype=torch.bool),
                torch.tensor([0,0,0,0,1,1,1,0],dtype=torch.bool),
                torch.tensor([0,0,0,0,0,0,0,1],dtype=torch.bool))
        counts=tuple(int(g.sum()) for g in groups)
        full=balanced_loss(value,groups,.5)
        parts=[balanced_loss(value[start:start+2],tuple(g[start:start+2] for g in groups),.5,
            population_counts=counts,population_size=8) for start in range(0,8,2)]
        torch.testing.assert_close(torch.stack(parts).mean(),full)
        torch.stack(parts).mean().backward()
        self.assertAlmostEqual(float(value.grad[-1]),.5)
        self.assertAlmostEqual(float(value.grad[4:7].sum()),.5,places=6)

    def test_absent_groups_are_finite_and_have_no_fabricated_replay_loss(self):
        groups=(torch.ones(4,dtype=torch.bool),torch.zeros(4,dtype=torch.bool),torch.zeros(4,dtype=torch.bool))
        value=torch.arange(4.).requires_grad_(True)
        result=balanced_loss(value,groups,.5);result.backward()
        self.assertEqual(float(result.detach()),1.5)
        torch.testing.assert_close(value.grad,torch.full((4,),.25))

    def test_advantages_normalized_within_task_groups(self):
        raw=torch.zeros(6,51);raw[2:,47]=1
        replay=torch.tensor([0,0,0,0,1,1],dtype=torch.bool)
        groups=masks(raw,replay)
        result=normalize_advantages(torch.tensor([-100.,-80.,-4.,-2.,100.,120.]),groups)
        for group in groups:
            self.assertAlmostEqual(float(result[group].mean()),0.)
            self.assertAlmostEqual(float(result[group].std(unbiased=False)),1.)

    def test_kl_schedule_never_exceeds_each_actors_initial_learning_rate(self):
        for initial in [1e-4,1e-5]:
            lr=initial
            for _ in range(100):lr=bounded_lr(lr,initial,0.,.01)
            self.assertEqual(lr,initial)
            for _ in range(100):lr=bounded_lr(lr,initial,1.,.01)
            self.assertGreaterEqual(lr,initial*.01)
            self.assertLessEqual(lr,initial)

    def test_invalid_stability_options_rejected(self):
        for value in [dict(throw_kl_limit=0),dict(replay_loss_weight=1),dict(reference_weight=-1),dict(typo=1)]:
            with self.assertRaises(ValueError):options(value)

if __name__=='__main__':unittest.main()
