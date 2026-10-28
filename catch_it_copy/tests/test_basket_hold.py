import unittest
from types import SimpleNamespace as N

import numpy as np
import torch

from gym_dcmm.utils import basket_mobile as mobile
from gym_dcmm.utils.basket_hold import preparation_progress, arm_drive_progress, arm_drive_reward
from gym_dcmm.utils.basket_transfer import migrate_actor
from gym_dcmm.algs.ppo_dcmm.models_track import ActorCritic
from gym_dcmm.algs.ppo_dcmm.utils import RunningMeanStd
from tests.test_basket_mobile import config, parking
from tests.test_basket_fixed_policy import SCOPE, ArrayTensor


class HoldTests(unittest.TestCase):
    def test_fixed_arm_drive_reward_is_bounded_single_release_and_collision_gated(self):
        state = dict(arm_drive_pending=2., last_contact_arm_drive=.75)
        self.assertEqual(arm_drive_reward(state, {}, 2.),
                         dict(arm_drive_progress=2., arm_drive_release=0.))
        self.assertEqual(arm_drive_reward(state, {'confirmed':True}, 2.)['arm_drive_release'], 1.5)
        self.assertEqual(arm_drive_reward(state, {'confirmed':True}, 2.)['arm_drive_release'], 0.)
        state = dict(arm_drive_pending=2., last_contact_arm_drive=1.)
        self.assertEqual(sum(arm_drive_reward(state, {'confirmed':True}, 2., True).values()), 0.)

    def test_arm_drive_needs_forward_transport_and_proximal_speed(self):
        state = {}
        def sample(position, arm_speed, ball_speed):
            arm_drive_progress(state, position, [0,arm_speed,0], [0,ball_speed,0], [0,1,0], 2.)
        sample([0,0,0], 0., 0.)
        sample([0,.12,0], 0., 1.)  # Wrist-only launch cannot earn arm credit.
        self.assertEqual(state['arm_drive_pending'], 0.)
        sample([0,.12,0], .6, 0.)  # Empty forward sweep cannot earn credit.
        self.assertEqual(state['arm_drive_pending'], 0.)
        sample([0,-.12,0], .6, .6)
        self.assertEqual(state['arm_drive_pending'], 0.)
        sample([0,.12,0], .6, .6)
        self.assertEqual(state['arm_drive_pending'], 2.)
        sample([0,0,0], 0., 0.)
        sample([0,.12,0], .6, .6)
        self.assertEqual(state['arm_drive_pending'], 2.)  # Repetition cannot farm credit.
        sample([0,.12,0], 0., 0.)
        self.assertEqual(state['last_contact_arm_drive'], 0.)

    def test_arm_drive_release_credit_is_qualified_once_and_collision_free(self):
        cfg = config(arm_drive_bonus=2.)
        for qualified, collision, expected in ((True,False,1.5),(False,False,0.),(True,True,0.)):
            state = mobile.new_state(0.)
            state.update(unlocked=True, release_qualified=qualified, last_contact_arm_drive=.75)
            def pay():
                return mobile.reward(state, {'confirmed':True, 'aim_paid':True}, parking(), False, False, 1., cfg,
                    robot_collision=collision, center=[0,1,0], gravity=[0,0,-9.81])[1]
            self.assertEqual(pay()['arm_drive_release'], expected)
            self.assertEqual(pay()['arm_drive_release'], 0.)

    def test_timed_hold_requires_contact_and_low_ball_speed(self):
        cfg=config(hold_only=True,fixed_throw_probability=1.,hold_time_range=[.3,.6])
        state=mobile.new_state(0.);state['hold_until']=.5
        launch={}
        for now,touch,speed,ready in ((.4,True,0.,False),(.5,False,0.,False),
                                      (.51,True,.3,False),(.52,True,.1,True)):
            mobile.update_physics(state,launch,parking(),[0.,0.,1.],[0.,speed,0.],
                [0.,1.,0.],now,.01,touch,False,cfg)
            self.assertEqual(state['unlocked'],ready)
        self.assertEqual(state['ready_pending'],0.)
        self.assertEqual(state['ready_time'],.52)
        self.assertEqual(state['held_ball_speed_at_ready'],.1)

    def test_hold_mode_base_command_is_always_zero(self):
        cfg=config(hold_only=True,fixed_throw_probability=1.,hold_time_range=[.3,.6])
        state=mobile.new_state(0.);state['physics_time']=1.
        np.testing.assert_array_equal(mobile.base_command(state,[1.,-1.],.04,cfg),[0.,0.])
        policy=N(mobile_basket=True,basket_training_stage='hold',action_catch_denorm=[1.5,.025,.15])
        actions=np.arange(20,dtype=float)[None]
        output=SCOPE['action2dict'](policy,ArrayTensor(actions))
        np.testing.assert_array_equal(output['base'],[[0.,0.]])
        np.testing.assert_allclose(output['arm'],actions[:,:6]*.025)
        np.testing.assert_allclose(output['hand'],actions[:,6:18]*.15)

    def test_preparation_credit_is_bounded_and_cannot_repeat(self):
        cfg=config(hold_aim_bonus=4.)
        state={}
        for velocity in ([0.,0.,0.],[0.,1.,0.],[0.,0.,0.],[0.,1.,0.]):
            preparation_progress(state,[0.,0.,1.],velocity,[0.,1.,0.],[0.,0.,-2.],cfg)
        self.assertGreater(state['held_aim_pending'],3.9)
        self.assertLessEqual(state['held_aim_pending'],4.)
        first=state['held_aim_pending']
        preparation_progress(state,[0.,0.,1.],[0.,1.,0.],[0.,1.,0.],[0.,0.,-2.],cfg)
        self.assertEqual(first,state['held_aim_pending'])

    def test_hold_options_reject_instant_or_random_target_hold_task(self):
        for options in (dict(hold_only=True),dict(hold_time_range=[-.1,.3]),
                        dict(hold_only=True,fixed_throw_probability=1.,hold_time_range=[.1,.3])):
            with self.assertRaises(ValueError):config(**options)

    def test_held_policy_reuse_does_not_shift_normalization_twice(self):
        def model(n):return ActorCritic(dict(actor_units=[16,8],actions_num=n,input_shape=(51,),separate_value_mlp=True))
        source,target=model(18),model(20)
        src_norm,dst_norm=RunningMeanStd((51,)).eval(),RunningMeanStd((51,)).eval()
        src_norm.running_mean.copy_(torch.linspace(-1.,1.,51))
        raw=torch.randn(5,51)
        expected=source.act_inference(dict(obs=src_norm(raw)))
        migrate_actor(target,dst_norm,dict(model=source.state_dict(),running_mean_std=src_norm.state_dict(),
            action_layout='held_arm6_hand12',basket_arm_control='joint_velocity',
            basket_training_stage='hold',basket_observation_version='mobile51_v1'))
        torch.testing.assert_close(target.act_inference(dict(obs=dst_norm(raw)))[:,2:],expected)
        torch.testing.assert_close(dst_norm.running_mean,src_norm.running_mean)


if __name__ == '__main__':unittest.main()
