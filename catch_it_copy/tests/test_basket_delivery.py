import unittest
from unittest.mock import patch
import numpy as np
from gym_dcmm.utils import basket_delivery as delivery
from tests.test_basket_mixed import environment


class DeliveryTests(unittest.TestCase):
    def setUp(self):self.options=delivery.make_options()

    def test_reachable_ballistics_beats_short_throw_without_prescribing_upward_velocity(self):
        center=np.array([0,1.05,.55]);gravity=np.array([0,0,-9.81]);p=np.array([0,.5,.7]);time=.35
        velocity=(center-p-.5*gravity*time*time)/time
        good,error=delivery.quality(p,velocity,center,gravity,25.,self.options)
        weak,_=delivery.quality(p,[0,.9,0],center,gravity,25.,self.options)
        self.assertAlmostEqual(good,1.)
        self.assertAlmostEqual(error,0.,places=6)
        self.assertGreater(good,weak)
        p=np.array([0,.5,1.]);time=.275
        velocity=(center-p-.5*gravity*time*time)/time
        self.assertLess(velocity[2],0.)
        q,_=delivery.quality(p,velocity,center,gravity,25.,self.options)
        self.assertAlmostEqual(q,1.)

    def test_idle_and_backward_motion_do_not_earn_guidance(self):
        for velocity in ([0,0,0],[0,-1,1]):
            q,_=delivery.quality([0,.5,.7],velocity,[0,1.05,.55],[0,0,-9.81],25.,self.options)
            self.assertEqual(q,0.)

    def test_guidance_bounded_and_cannot_be_farmed_with_repeated_swings(self):
        state=delivery.new_state(.1)
        for q in (.4,.2,.4,.8,.5,1.):delivery.observe(state,q,.2,self.options)
        self.assertAlmostEqual(state['pending'],.9*self.options['progress_bonus'])
        terms=delivery.adjustments(state,{},self.options,done=False,success=False,collision=False)
        self.assertAlmostEqual(terms['ballistic_progress'],7.2)
        delivery.observe(state,1.,0.,self.options)
        self.assertEqual(delivery.adjustments(state,{},self.options,done=False,success=False,collision=False)['ballistic_progress'],0.)

    def test_carry_scales_with_quality_but_safety_and_slip_not_discounted(self):
        state=delivery.new_state(.2);state['release_quality']=.4
        original=dict(carry_forward=2.,carry_position=1.,carry_release=2.,carry_slip=-2.,robot_collision=-40.,valid_launch=5.,flight_progress=5.)
        terms=delivery.adjustments(state,original,self.options,done=False,success=False,collision=False)
        self.assertAlmostEqual(terms['carry_forward'],.8)
        self.assertAlmostEqual(terms['carry_release'],.8)
        self.assertEqual(terms['valid_launch'],2.)
        self.assertEqual(terms['flight_progress'],3.)
        self.assertNotIn('robot_collision',terms)
        self.assertNotIn('carry_slip',terms)

    def test_miss_is_paid_once_and_true_success_is_not_penalized(self):
        state=delivery.new_state()
        self.assertEqual(delivery.adjustments(state,{},self.options,done=True,success=False,collision=False)['task_miss'],-8.)
        self.assertEqual(delivery.adjustments(state,{},self.options,done=True,success=False,collision=False)['task_miss'],0.)
        self.assertEqual(delivery.adjustments(delivery.new_state(),{},self.options,done=True,success=True,collision=False)['task_miss'],0.)

    def test_bad_options_rejected(self):
        for options in ({'typo':1},{'aim_sigma':0},{'miss_cost':-1},{'launch_scale':2}):
            with self.assertRaises(ValueError):delivery.make_options(options)

    def test_legacy_weights_cannot_silently_use_new_reward_preset(self):
        from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_basket_mixed import PPO_Basket_Mixed
        agent=PPO_Basket_Mixed.__new__(PPO_Basket_Mixed)
        agent.device='cpu';agent.delivery_version='ballistic_v2'
        checkpoint=dict(basket_mixed_protocol='mixed37_known_radius_mass_v1')
        with patch('torch.load',return_value=checkpoint):
            with self.assertRaisesRegex(ValueError,'reward preset'):
                agent.restore_test('/unused/old.pth')

    def test_single_ball_best_success_does_not_wait_for_other_balls(self):
        from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_basket_mixed import PPO_Basket_Mixed
        from gym_dcmm.algs.ppo_dcmm.utils import AverageScalarMeter
        agent=PPO_Basket_Mixed.__new__(PPO_Basket_Mixed)
        agent.ball_mode='tennis_ball';agent.per_ball=[AverageScalarMeter(200) for _ in range(3)]
        self.assertIs(agent.selection_success_meter(),agent.per_ball[1])


class DeliveryEnvironmentTests(unittest.TestCase):
    def test_returned_reward_terms_and_totals_match_and_reset_has_no_free_credit(self):
        env=environment(basket_mixed_ball='original',basket_delivery_version='ballistic_v2')
        try:
            env.reset(seed=123)
            self.assertEqual(env.delivery_state['pending'],0.)
            total=0.
            for step in range(300):
                obs,reward,done,trunc,info=env.step(dict(base=np.zeros(2),arm=np.zeros(6),hand=np.zeros(12)))
                total+=reward
                self.assertAlmostEqual(reward,sum(info['basket_reward_terms'].values()),places=6)
                self.assertAlmostEqual(total,sum(info['basket_reward_totals'].values()),places=6)
                if done or trunc:
                    self.assertEqual(info['basket_reward_totals']['task_miss'],-8.)
                    break
            else:self.fail('Episode did not terminate')
            env.reset(seed=123)
            self.assertFalse(env.delivery_state['miss_paid'])
            self.assertEqual(env.delivery_state['pending'],0.)
        finally:env.close()

if __name__=='__main__':unittest.main()
