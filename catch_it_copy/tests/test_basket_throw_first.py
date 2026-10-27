import copy
import unittest

import numpy as np
import torch

from gym_dcmm.utils import basket_mobile as mobile
from gym_dcmm.utils.basket_policy_mask import masked_mean, clear_actor_gradients
from gym_dcmm.algs.ppo_dcmm.models_track import ActorCritic
from tests.test_basket_mobile import config, parking


class ThrowFirstTests(unittest.TestCase):
    def test_contact_flicker_grace_keeps_settling_but_never_confirms_early_throw(self):
        for grace, expected in ((0., False), (.03, True)):
            cfg = config(settle_contact_grace=grace)
            state, launch = mobile.new_state(.5), {}
            for i in range(10):
                mobile.update_physics(state,launch,parking(), [0.,.4,1.], [0.,1.,0.],
                    [0.,1.,0.], .4+i*.02, .02, i != 5, False, cfg)
            self.assertEqual(state['unlocked'], expected)
            self.assertFalse(launch.get('confirmed', False))

    def test_post_ready_deadline_gives_late_arrival_its_full_throw_budget(self):
        cfg = config(post_ready_seconds=2.5)
        state = mobile.new_state(.5)
        self.assertEqual(mobile.episode_deadline(state,cfg), 8.)
        state.update(unlocked=True, ready_time=7.9)
        self.assertAlmostEqual(mobile.episode_deadline(state,cfg), 10.4)
        obs = mobile.observation(state,np.zeros(6),np.zeros(6),8.,cfg)
        self.assertAlmostEqual(obs[-1],.96)

    def test_position_brake_has_hysteresis_and_does_not_choose_drive_direction(self):
        cfg = config(hold_position=True)
        state = mobile.new_state(.5);state['physics_time']=1.
        p = parking(.03);p.update(in_position=True,error_world=np.array([.03,0.]))
        np.testing.assert_array_equal(mobile.base_command(state,[1.,0.],.04,cfg,p), [0.,0.])
        p.update(in_position=False,error_world=np.array([.12,0.]))
        np.testing.assert_array_equal(mobile.base_command(state,[1.,0.],.04,cfg,p), [0.,0.])
        p['error_world'] = np.array([.16,0.])
        self.assertGreater(mobile.base_command(state,[1.,0.],.04,cfg,p)[0],0.)

    def test_release_margin_requires_prior_unlock_and_bounds_recoil(self):
        cfg = config(release_speed_tolerance=.15,release_position_tolerance=.10)
        for ready, speed, qualified in ((False,.12,False),(True,.12,True),(True,.16,False)):
            state, launch = mobile.new_state(.5), {}
            state['unlocked'] = ready
            p = parking(.02,False,speed);p['error_world']=np.array([0.,.02])
            mobile.update_physics(state,launch,p,[0.,.4,1.],[0.,1.,0.],[0.,1.,0.],1.,.02,True,False,cfg)
            self.assertEqual(state['last_contact_qualified'],qualified)

    def test_no_contact_grace_or_replay_probability_out_of_bounds(self):
        for options in (dict(settle_contact_grace=.05),dict(fixed_throw_probability=1.1),
                        dict(fixed_throw_probability=-.1),dict(parking_only=True,fixed_throw_probability=.3)):
            with self.assertRaises(ValueError):
                config(**options)

    def test_masked_actor_does_not_move_from_adam_momentum(self):
        torch.manual_seed(1)
        model = ActorCritic(dict(actor_units=[8],actions_num=18,input_shape=(51,),separate_value_mlp=True))
        optimizer = torch.optim.Adam(model.parameters(),lr=.01)
        x=torch.randn(4,51)
        mu,_,v=model._actor_critic(dict(obs=x))
        (mu.square().mean()+v.square().mean()+model.sigma.square().mean()).backward();optimizer.step()
        actor_before={k:v.clone() for k,v in model.state_dict().items() if not k.startswith(('value.','value_mlp.'))}
        critic_before=model.value.weight.detach().clone()
        optimizer.zero_grad();mu,_,v=model._actor_critic(dict(obs=x))
        loss=masked_mean(mu.square().mean(-1),torch.zeros(4,dtype=torch.bool))+v.square().mean()
        loss.backward();clear_actor_gradients(model);optimizer.step()
        for key,value in actor_before.items():
            torch.testing.assert_close(model.state_dict()[key],value,rtol=0,atol=0)
        self.assertFalse(torch.equal(critic_before,model.value.weight))

    def test_masked_mean_ignores_unexecuted_actions(self):
        loss=torch.tensor([100.,2.,4.],requires_grad=True)
        mean=masked_mean(loss,torch.tensor([False,True,True]));mean.backward()
        self.assertEqual(mean.item(),3.)
        torch.testing.assert_close(loss.grad,torch.tensor([0.,.5,.5]))


if __name__ == '__main__':
    unittest.main()
