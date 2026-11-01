import copy
import unittest
from types import SimpleNamespace as N

import numpy as np
import torch

from gym_dcmm.algs.ppo_dcmm.models_track import ActorCritic
from gym_dcmm.algs.ppo_dcmm.utils import RunningMeanStd
from gym_dcmm.utils.basket_throw_policy import FrozenThrowPolicy
from gym_dcmm.utils.basket_fingers import make_options as finger_options
from gym_dcmm.utils.basket_carry import make_options as carry_options
from gym_dcmm.utils import basket_mobile as mobile
from tests.test_basket_mobile import config, parking
from tests.test_basket_fixed_policy import SCOPE, ArrayTensor


class Mobile41Tests(unittest.TestCase):
    def setUp(self):
        self.source = ActorCritic(dict(actor_units=[16,8], actions_num=18,
            input_shape=(35,), separate_value_mlp=True))
        self.norm = RunningMeanStd((35,)).eval()
        self.norm.running_mean.copy_(torch.randn(35))
        self.norm.running_var.copy_(torch.rand(35)+.5)
        self.checkpoint = dict(model=self.source.state_dict(),running_mean_std=self.norm.state_dict(),
            action_layout='arm6_hand12',basket_arm_control='joint_velocity',
            basket_fixed_protocol='aim_carry_fingers_v1',
            basket_finger_options=finger_options(),basket_carry_options=carry_options())
        self.frozen = FrozenThrowPolicy(self.checkpoint,finger_options(),carry_options(),'cpu')

    def test_exact_source_actions_and_normalization_preserved_at_throw_stand(self):
        raw=torch.randn(4,51);raw[:,47]=1.
        fixed=raw[:,:35].clone();fixed[:,34]+=.2
        expected=self.source.act_inference(dict(obs=self.norm(fixed))).clamp(-1,1)
        torch.testing.assert_close(self.frozen.actions(raw),expected)
        self.assertTrue(all(not p.requires_grad for p in self.frozen.model.parameters()))
        self.assertTrue(all(not p.requires_grad for p in self.frozen.normalizer.parameters()))
        snapshot=copy.deepcopy(self.frozen.normalizer.state_dict())
        self.frozen.actions(raw)
        for k,v in snapshot.items():torch.testing.assert_close(self.frozen.normalizer.state_dict()[k],v)

    def test_no_throw_while_driving_and_embedded_checkpoint_round_trip(self):
        raw=torch.randn(4,51);raw[:,47]=torch.tensor([0.,1.,0.,1.])
        actions=self.frozen.actions(raw)
        torch.testing.assert_close(actions[[0,2]],torch.zeros(2,18))
        loaded=FrozenThrowPolicy(self.frozen.export(),finger_options(),carry_options(),'cpu')
        torch.testing.assert_close(actions,loaded.actions(raw))

    def test_rejects_wrong_action_semantics_or_reward_contract(self):
        for change in [dict(action_layout='base2_arm6_hand12'),dict(basket_arm_control='cartesian'),
                       dict(basket_fixed_protocol='aim_carry_v1'),dict(basket_finger_options={})]:
            with self.assertRaises(ValueError):
                FrozenThrowPolicy(dict(self.checkpoint,**change),finger_options(),carry_options(),'cpu')

    def test_base_and_frozen_hand_channels_are_independent(self):
        raw=torch.randn(2,51);raw[:,47]=1.
        actor=N(mobile_basket=True,basket_training_stage='navigate_throw',mobile_base_scale=.3,
            throw_policy=self.frozen,obs=dict(obs=raw),action_catch_denorm=[1.5,.025,.15])
        requested=np.ones((2,20))*.4
        result=SCOPE['action2dict'](actor,ArrayTensor(requested))
        frozen=self.frozen.actions(raw).numpy()
        np.testing.assert_allclose(result['base'],.12)
        np.testing.assert_allclose(result['arm'],frozen[:,:6]*.025)
        np.testing.assert_allclose(result['hand'],frozen[:,6:]*.15)

    def test_stop_then_settle_required_before_unlock_and_base_braked_after(self):
        cfg=config(preserve_fixed_throw=True,brake_after_unlock=True,carry_settle_seconds=.1,
                   settle_seconds=.3)
        state=mobile.new_state(.5);launch={}
        moving=parking();moving.update(settled=False,speed=.2)
        for i in range(300):
            mobile.update_physics(state,launch,moving,[0,0,.7],[0,0,0],[0,1,.55],
                i*.002,.002,True,False,cfg)
        self.assertFalse(state['unlocked'])
        for i in range(151):
            mobile.update_physics(state,launch,parking(),[0,0,.7],[0,0,0],[0,1,.55],
                .6+i*.002,.002,True,False,cfg)
        self.assertTrue(state['unlocked'])
        np.testing.assert_array_equal(mobile.base_command(state,[.3,.3],.04,cfg),[0,0])

    def test_parking_brake_does_not_latch_outside_unlock_position_tolerance(self):
        cfg=config(preserve_fixed_throw=True,hold_position=True,carry_settle_seconds=.1)
        state=mobile.new_state(.5);state['physics_time']=1.
        inside=parking();inside.update(in_position=True,error_world=np.array([0.,0.]))
        mobile.base_command(state,[.2,0],.04,cfg,inside)
        self.assertTrue(state['parking_brake_latched'])
        outside=parking();outside.update(in_position=False,
            error_world=np.array([1.1*cfg.basket_track_position_tolerance,0.]))
        command=mobile.base_command(state,[.2,0],.04,cfg,outside)
        self.assertFalse(state['parking_brake_latched'])
        self.assertGreater(command[0],0.)

    def test_navigation_slip_cost_drained_once_and_fixed_hit_bonus_retained(self):
        state=mobile.new_state(.5);state.update(unlocked=True,release_qualified=True,
            carry_slip_pending=-.2)
        launch=dict(confirmed=True,candidate=dict(position=[0,0,.8],velocity=[0,2,1]))
        cfg=config(preserve_fixed_throw=True)
        _,terms=mobile.reward(state,launch,parking(),False,False,.1,cfg,
            center=[0,1,.55],gravity=[0,0,-9.81],hit=True)
        self.assertEqual(terms['basket_hit'],20.)
        self.assertEqual(terms['navigation_slip'],-.2)
        _,again=mobile.reward(state,launch,parking(),False,False,.2,cfg,
            center=[0,1,.55],gravity=[0,0,-9.81],hit=True)
        self.assertEqual(again['navigation_slip'],0.)
        self.assertEqual(again['basket_hit'],0.)

if __name__=='__main__':unittest.main()
