import ast
from pathlib import Path
from types import SimpleNamespace as N
import unittest

import numpy as np

from gym_dcmm.utils import basket_carry as carry


class CarryTests(unittest.TestCase):
    def setUp(self):
        self.options = carry.make_options()
        self.local = np.array([.003, .054, .091])
        self.state = carry.new_state([0.,0.,0.], self.local)

    def move(self, distance, local=None, speed=.3):
        carry.update(self.state, [0.,distance,0.], [0.,speed,0.], [0.,.6,0.],
                     self.local if local is None else local, [0.,1.,0.], self.options)

    def test_stationary_holding_and_repeated_sweeps_cannot_farm_reward(self):
        for _ in range(100):
            self.move(0., speed=0.)
        self.assertEqual(sum(carry.reward(self.state, {}, self.options).values()), 0.)
        total = 0.
        for _ in range(20):
            self.move(.06)
            total += sum(carry.reward(self.state, {}, self.options).values())
            self.move(0., speed=-.3)
            total += sum(carry.reward(self.state, {}, self.options).values())
        self.assertAlmostEqual(total, 8.)
        self.assertAlmostEqual(carry.reward(self.state, {'confirmed':True}, self.options)['carry_release'], 0.)

    def test_stable_ball_is_preferred_to_sliding_ball_at_same_arm_motion(self):
        self.move(.03)
        good = carry.reward(self.state, {}, self.options)
        self.state = carry.new_state([0.,0.,0.], self.local)
        self.move(.03, local=self.local+[0.,.06,0.])
        bad = carry.reward(self.state, {}, self.options)
        self.assertAlmostEqual(good['carry_forward'], bad['carry_forward'])
        self.assertGreater(good['carry_position'], bad['carry_position'])
        self.assertLess(bad['carry_slip'], 0.)
        self.assertGreater(sum(good.values()), sum(bad.values()))

    def test_small_forward_motion_has_signal_and_release_is_once_and_bounded(self):
        self.move(.01)
        terms = carry.reward(self.state, {}, self.options)
        self.assertGreater(terms['carry_forward'], 0.)
        self.move(.06)
        terms2 = carry.reward(self.state, {'confirmed':True}, self.options)
        self.assertAlmostEqual(sum(terms.values())+sum(terms2.values()), 12.)
        self.assertEqual(sum(carry.reward(self.state, {'confirmed':True}, self.options).values()), 0.)

    def test_no_proximal_motion_or_robot_collision_cannot_earn_bonus(self):
        self.move(.06, speed=0.)
        self.assertEqual(sum(carry.reward(self.state, {'confirmed':True}, self.options).values()), 0.)
        self.state = carry.new_state([0.,0.,0.], self.local)
        self.move(.06)
        self.assertEqual(sum(carry.reward(self.state, {'confirmed':True}, self.options, collision=True).values()), 0.)

    def test_actual_environment_keeps_full_basket_credit_for_short_throw(self):
        from gym_dcmm.utils.basket_aim import make_config
        from gym_dcmm.utils.basket_simple import reward as simple_launch_reward
        import configs.env.DcmmCfg as base
        path = Path(__file__).resolve().parents[1]/'gym_dcmm/envs/DcmmVecEnv.py'
        tree = ast.parse(path.read_text())
        cls = next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='DcmmVecEnv')
        method = next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='compute_reward')
        scope = dict(np=np, DcmmCfg=base, simple_launch_reward=simple_launch_reward)
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),scope)
        state = dict(confirmed=True, candidate=dict(position=np.array([0.,.5,.65]),
            velocity=np.array([0.,1.6,1.4])), carry=self.state)
        env = N(object_motion='throw_basket',basket_reward_version='basket_aim_v1',
            basket_cfg=make_config(base),terminated=True,terminated_reason='basket_score',
            basket_simple_state=state,basket_reward_totals={},basket_carry_reward=True,
            basket_carry_options=self.options,basket_fixed_physical=True,
            basket_preparation={},basket_spawn_check={},start_time=0.,basket_last_contact=None,
            basket_aim_hit_time=.5,basket_center=np.array([0.,1.05,.55]),basket_phase='flight',
            Dcmm=N(data=N(qpos=np.zeros(44),time=1.),model=N(opt=N(gravity=np.array([0.,0.,-9.81])))))
        info=dict(env_time=1.,success=True)
        value=scope['compute_reward'](env,{},info,dict(arm=np.zeros(6)))
        self.assertEqual(info['basket_reward_terms']['basket_hit'],20.)
        self.assertTrue(info['safe_basket_success'])
        self.assertNotIn('physical_release_quality',info['basket_control'])
        self.assertAlmostEqual(value,sum(info['basket_reward_terms'].values()))


if __name__=='__main__':
    unittest.main()
