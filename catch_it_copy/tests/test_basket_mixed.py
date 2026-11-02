import copy
import unittest
import numpy as np
import torch

from gym_dcmm.algs.ppo_dcmm.models_track import ActorCritic
from gym_dcmm.algs.ppo_dcmm.utils import RunningMeanStd,AverageScalarMeter
from gym_dcmm.algs.ppo_dcmm.basket_mixed_policy import MixedNormalizer,import_throw,stratified_mean,MacroSuccess
from gym_dcmm.utils.basket_fingers import make_options as finger_options
from gym_dcmm.utils.basket_carry import make_options as carry_options
from gym_dcmm.envs.basket_mixed import BasketMixedEnv,make_options


def environment(**kwargs):
    return BasketMixedEnv(task='Catching',object_motion='throw_basket',basket_fixed_base_training=True,
        basket_fixed_physical=True,basket_carry_reward=True,basket_finger_reward=True,
        basket_reward_version='basket_aim_v1',basket_arm_control='joint_velocity',
        basket_fixed_hand_flex=[.8,.7,.7],render_mode=None,viewer=False,imshow_cam=False,**kwargs)


class MixedPolicyTests(unittest.TestCase):
    def source(self):
        model=ActorCritic(dict(actor_units=[16,8],actions_num=18,input_shape=(35,),separate_value_mlp=True))
        norm=RunningMeanStd((35,)).eval()
        return model,norm,dict(model=model.state_dict(),running_mean_std=norm.state_dict(),
            basket_fixed_protocol='aim_carry_fingers_v1',basket_arm_control='joint_velocity',action_layout='arm6_hand12',
            basket_finger_options=finger_options(),basket_carry_options=carry_options())

    def test_fixed_transfer_preserves_mean_std_and_adds_trainable_property_inputs(self):
        source,norm,checkpoint=self.source()
        target=ActorCritic(dict(actor_units=[16,8],actions_num=18,input_shape=(37,),separate_value_mlp=True))
        normalizer=MixedNormalizer().eval()
        import_throw(target,normalizer,checkpoint,finger_options(),carry_options())
        raw=torch.randn(4,35);extended=torch.cat((raw,torch.tensor([[.04,.05]]*4)),dim=-1)
        torch.testing.assert_close(source.act_inference(dict(obs=norm(raw))),target.act_inference(dict(obs=normalizer(extended))))
        torch.testing.assert_close(source.sigma,target.sigma)
        self.assertTrue(all(p.requires_grad for p in target.actor_mlp.parameters()))
        normalized=normalizer(extended);normalized[:,-2:]=torch.tensor([-.6,.9])
        target(dict(obs=normalized,prev_actions=torch.ones(4,18)*.2))['prev_neglogp'].mean().backward()
        self.assertGreater(float(target.actor_mlp.mlp[0].weight.grad[:,-2:].abs().sum()),0.)

    def test_mobile_joint_transfer_extracts_throw_only(self):
        source,norm,fixed=self.source();state=fixed['model']
        mobile=dict(fixed,action_layout='joint_base2_arm18_phase_v1')
        mobile['model']={**{'throw_actor.'+k[len('actor_mlp.'):]:v for k,v in state.items() if k.startswith('actor_mlp.')},
            **{'throw_mu.'+k[len('mu.'):]:v for k,v in state.items() if k.startswith('mu.')},'throw_sigma':state['sigma']}
        mobile['running_mean_std']={'throwing.'+k:v for k,v in norm.state_dict().items()}
        target=ActorCritic(dict(actor_units=[16,8],actions_num=18,input_shape=(37,),separate_value_mlp=True))
        normalizer=MixedNormalizer().eval()
        self.assertEqual(import_throw(target,normalizer,mobile,finger_options(),carry_options()),'mobile_joint_throw35')
        raw=torch.randn(3,35);extended=torch.cat((raw,torch.tensor([[.04,.05]]*3)),dim=-1)
        torch.testing.assert_close(source.act_inference(dict(obs=norm(raw))),target.act_inference(dict(obs=normalizer(extended))))

    def test_physical_input_scaling_and_statistics_stay_fixed(self):
        norm=MixedNormalizer();snapshot=copy.deepcopy(norm.state_dict())
        norm.train();raw=torch.zeros(3,37);raw[:,-2:]=torch.tensor([[.04,.05],[.033,.058],[.036,.138]])
        value=norm(raw)
        torch.testing.assert_close(value[0,-2:],torch.zeros(2))
        self.assertLess(float(value[1,-2]),0.);self.assertGreater(float(value[2,-1]),1.)
        for k,v in snapshot.items():torch.testing.assert_close(norm.state_dict()[k],v)

    def test_per_ball_rollout_loss_not_biased_by_episode_lengths(self):
        labels=torch.tensor([0]*20+[1]*2+[2]*2);counts=(20,2,2)
        values=torch.tensor([1.]*20+[3.]*2+[5.]*2,requires_grad=True)
        loss=stratified_mean(values,labels,counts,24)
        self.assertAlmostEqual(float(loss.detach()),3.)
        chunks=[stratified_mean(values[i:i+6],labels[i:i+6],counts,24) for i in range(0,24,6)]
        torch.testing.assert_close(torch.stack(chunks).mean(),loss)

    def test_macro_success_requires_all_balls_then_weights_them_equally(self):
        meters=[AverageScalarMeter(200) for _ in range(3)];macro=MacroSuccess(meters)
        meters[0].update(torch.ones(20));self.assertEqual(len(macro),0)
        meters[1].update(torch.zeros(20));meters[2].update(torch.ones(20))
        self.assertEqual(len(macro),60);self.assertAlmostEqual(macro.get_mean(),2/3)

    def test_invalid_sampling_and_grip_options_rejected(self):
        for value in [dict(weights=[0,0,0]),dict(weights=[1,-1,1]),dict(tennis_hand=[1,2]),dict(unknown=1)]:
            with self.assertRaises(ValueError):make_options(value)


class MixedEnvironmentTests(unittest.TestCase):
    def test_switching_balls_restores_original_dynamics_and_provides_correct_observation_metadata(self):
        env=environment()
        try:
            original=copy.deepcopy(env._original)
            for name in ('original','tennis_ball','baseball','original'):
                env.ball_mode=name;obs,info=env.reset(seed=123)
                self.assertEqual(env.obs_c_dim,37)
                self.assertEqual(obs['basket_ball_id'][0],env.ball_index)
                self.assertEqual(info['basket_ball'],name)
                model=env.Dcmm.model;body=env._ball_body
                self.assertAlmostEqual(float(obs['basket_ball_properties'][0]),model.geom_size[env.object_id,0],places=7)
                self.assertAlmostEqual(float(obs['basket_ball_properties'][1]),model.body_mass[body],places=7)
                self.assertGreaterEqual(env.basket_preparation['stable_seconds'],.1)
                self.assertEqual(model.geom_type[env.object_id],2) # MuJoCo sphere
                if name!='original':
                    profile=env.profiles[name]
                    np.testing.assert_allclose(model.body_inertia[body],profile['inertia_factor']*profile['mass_kg']*profile['radius_m']**2)
                else:
                    np.testing.assert_array_equal(model.body_inertia[body],original['inertia'])
                    np.testing.assert_array_equal(model.geom_friction[env.object_id],original['friction'])
                    np.testing.assert_array_equal(model.dof_damping[env._ball_dofs],original['damping'])
        finally:env.close()

    def test_sampling_is_seed_reproducible_without_modifying_other_environments(self):
        env=environment()
        try:
            first=[];second=[]
            for seed in range(6):env.reset(seed=seed);first.append(env.ball_index)
            for seed in range(6):env.reset(seed=seed);second.append(env.ball_index)
            self.assertEqual(first,second)
            self.assertGreater(len(set(first)),1)
        finally:env.close()

if __name__=='__main__':unittest.main()
