import unittest
from types import SimpleNamespace
from gym_dcmm.utils.basket_simple import update_launch, reward


CFG = SimpleNamespace(basket_simple_free_seconds=.05, basket_simple_forward_speed=.3,
                      basket_simple_target_distance=.5, basket_simple_launch_bonus=5.,
                      basket_simple_progress_bonus=10., basket_simple_failure_cost=5.)


class SimpleLaunchTests(unittest.TestCase):
    def step(self, state, time, x=0., speed=1., touching=False, collision=False):
        update_launch(state, [x, 0, 1], [speed, 0, 0], [2, 0, 1],
                      time, touching, collision, CFG)

    def test_robot_collision_penalty_after_launch_and_only_once(self):
        cfg = SimpleNamespace(**vars(CFG), basket_simple_collision_cost=20.)
        s = {'confirmed': True, 'launch_pending': 5., 'progress_pending': 10.}
        value, terms = reward(s, True, cfg, robot_collision=True)
        self.assertEqual(value, -5.)
        self.assertEqual(terms['robot_collision'], -20.)
        self.assertEqual(reward(s, True, cfg, robot_collision=True)[0], 0.)
        self.assertEqual(reward({}, True, cfg, robot_collision=True)[0], -25.)
        self.assertEqual(reward({'confirmed': True}, True, cfg)[0], 0.)

    def test_smoothness_normalized_action_and_reset(self):
        cfg = SimpleNamespace(**vars(CFG), basket_simple_smooth_weight=.01)
        s = {}
        self.assertEqual(reward(s, False, cfg, arm_action=[-1.]*6)[0], 0.)
        self.assertAlmostEqual(reward(s, False, cfg, arm_action=[1.]*6)[0], -.04)
        self.assertEqual(reward(s, False, cfg, arm_action=[1.]*6)[0], 0.)
        self.assertEqual(reward({}, False, cfg, arm_action=[1.]*6)[0], 0.)

    def test_self_contacts_exclude_adjacent_unloaded_and_separated(self):
        import numpy as np
        from unittest.mock import patch
        from gym_dcmm.utils.basket_simple import arm_self_contacts
        m = SimpleNamespace(geom_bodyid=[1,2,3,4], body_parentid=[0,0,1,2,0],
            body=lambda i: SimpleNamespace(name=['world','link1','link2','link3','object'][i]))
        contacts = [SimpleNamespace(geom1=0,geom2=b,dist=dist)
                    for b, dist in [(1,-.001),(2,-.001),(3,-.001),(2,.001),(2,-.001)]]
        d = SimpleNamespace(contact=contacts)
        fake = SimpleNamespace(mj_contactForce=lambda m,d,i,f: f.__setitem__(0, 10. if i != 4 else 0.))
        with patch.dict('sys.modules', {'mujoco':fake}):
            hits = arm_self_contacts(m,d,.1)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]['pair'], ['link1','link3'])

    def test_actual_environment_reward_branch_uses_ctrl(self):
        import ast
        from pathlib import Path
        import numpy as np
        source = Path(__file__).resolve().parents[1] / 'gym_dcmm/envs/DcmmVecEnv.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DcmmVecEnv')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_reward')
        scope = dict(np=np, simple_launch_reward=reward,
                     DcmmCfg=SimpleNamespace(**vars(CFG), basket_max_time=4.,
                         basket_simple_collision_cost=20., basket_simple_smooth_weight=.01))
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), scope)
        env = SimpleNamespace(object_motion='throw_basket', basket_reward_version='launch_simple_v1',
            terminated=False, basket_simple_state={}, basket_reward_totals={},
            basket_center=np.zeros(3), basket_phase='preparing',
            Dcmm=SimpleNamespace(data=SimpleNamespace(qpos=np.zeros(44))))
        info = dict(env_time=.04, success=False)
        run = scope['compute_reward']
        self.assertEqual(run(env, {}, info, {'arm': np.full(6, -.025)}), 0.)
        self.assertAlmostEqual(run(env, {}, info, {'arm': np.full(6, .025)}), -.04)
        env.terminated = True
        env.basket_arm_self_collision = True
        self.assertAlmostEqual(run(env, {}, info, {'arm': np.full(6, .025)}), -25.)
        self.assertFalse(info['safe_launch_success'])
        self.assertTrue(info['robot_dangerous_collision'])

    def test_confirmation_and_capped_total(self):
        s = {}
        self.step(s, 0, touching=True)
        self.step(s, .04, .04)
        self.assertEqual(reward(s, False, CFG)[0], 0.)
        self.step(s, .06, .1)
        self.assertAlmostEqual(reward(s, False, CFG)[0], 7.)
        self.step(s, .1, .6)
        self.assertAlmostEqual(reward(s, False, CFG)[0], 8.)
        self.step(s, .2, 1.)
        self.assertEqual(reward(s, True, CFG)[0], 0.)

    def test_drop_backward_and_no_prior_contact(self):
        for speed in (0., -.5, .29):
            s = {}
            self.step(s, 0, speed=speed, touching=True)
            self.step(s, .1, .1)
            self.assertEqual(reward(s, True, CFG)[0], -5.)
            self.assertEqual(reward(s, True, CFG)[0], 0.)
        s = {}
        self.step(s, .1, .5)
        self.assertEqual(reward(s, True, CFG)[0], -5.)

    def test_collision_before_confirmation(self):
        s = {}
        self.step(s, 0, touching=True)
        self.step(s, .02, .02, collision=True)
        self.step(s, .1, .1)
        self.assertEqual(reward(s, True, CFG)[0], -5.)

    def test_recontact_or_collision_stops_credit(self):
        for contact in ('touching', 'collision'):
            s = {}
            self.step(s, 0, touching=True)
            self.step(s, .06, .1)
            self.assertEqual(reward(s, False, CFG)[0], 7.)
            self.step(s, .1, .2, **{contact: True})
            self.step(s, .2, .6)
            self.assertEqual(reward(s, True, CFG)[0], 0.)

    def test_retreat_does_not_pay_twice(self):
        s = {}
        self.step(s, 0, touching=True)
        self.step(s, .06, .2)
        self.assertEqual(reward(s, False, CFG)[0], 9.)
        self.step(s, .1, .1)
        self.step(s, .2, .2)
        self.assertEqual(reward(s, False, CFG)[0], 0.)


if __name__ == '__main__':
    unittest.main()
