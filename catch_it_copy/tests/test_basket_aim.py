import unittest
from types import SimpleNamespace as N
import numpy as np
from gym_dcmm.utils.basket_aim import make_config, crossing_error, reward

class AimTests(unittest.TestCase):
    def config(self):
        return make_config(N(basket_simple_launch_bonus=5., basket_simple_failure_cost=5.,
            basket_simple_smooth_weight=.01, basket_tilt_deg=0.))

    def test_intersection_direction(self):
        self.assertAlmostEqual(crossing_error([0,0,1],[0,1,0],[0,1,0],[0,0,-2],0),0.)
        self.assertIsNone(crossing_error([0,0,0],[0,0,-1],[0,0,1],[0,0,-2],0))

    def test_once_and_collision(self):
        cfg=self.config()
        s=dict(confirmed=True, launch_pending=5., progress_pending=5.,
               candidate=dict(position=[0,0,1],velocity=[0,1,0]))
        args=dict(center=[0,1,0],gravity=[0,0,-2])
        self.assertAlmostEqual(reward(s,False,cfg,hit=True,**args)[0],40.)
        self.assertEqual(reward(s,False,cfg,hit=True,**args)[0],0.)
        self.assertEqual(reward(s,True,cfg,robot_collision=True,hit=True,**args)[0],-40.)

    def test_no_intersection_no_aim_reward(self):
        s=dict(confirmed=True,candidate=dict(position=[0,0,0],velocity=[0,0,-1]))
        value,terms=reward(s,False,self.config(),center=[0,0,1],gravity=[0,0,-2])
        self.assertEqual(terms['aim'],0.)

    def test_config_does_not_mutate_launch(self):
        base=N(basket_simple_progress_bonus=10.,basket_simple_collision_cost=20.)
        cfg=make_config(base)
        self.assertEqual(base.basket_simple_progress_bonus,10.)
        self.assertEqual(cfg.basket_simple_progress_bonus,5.)
