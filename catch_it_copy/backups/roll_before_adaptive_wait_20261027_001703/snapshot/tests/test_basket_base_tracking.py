"""CPU checks of actual control code, observation packing and policy transfer."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PARK = load_file('basket_parking', 'gym_dcmm/utils/basket_tracking.py')
CFG = load_file('basket_cfg', 'configs/env/DcmmCfg.py')
# Existing geometry fixtures exercise the optional basket-relative parking mode.
CFG.basket_track_forward_offset = None


def method(path, cls, name):
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name)


def execute(node, scope):
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual code>', 'exec'), scope)
    return scope[node.name]


class BasketTests(unittest.TestCase):
    def test_initial_position_target_is_fixed_as_robot_moves(self):
        cfg = SimpleNamespace(**vars(CFG))
        cfg.basket_track_forward_offset = .2
        initial = np.array([0., .118, .4])
        basket = np.array([.8, 2.2, .9])
        first = PARK.parking_state(initial, [0., 0.], 0., basket, cfg, initial)
        moved = PARK.parking_state(np.array([.8, .318, .4]), [0., 0.], 0., basket, cfg, initial)
        np.testing.assert_allclose(first['target'], [.8, .318, .9])
        np.testing.assert_allclose(first['target'], moved['target'])
        self.assertTrue(moved['settled'])
        with self.assertRaises(ValueError):
            PARK.parking_state(initial, [0., 0.], 0., basket, cfg)

    def test_tracking_holds_arm_and_hand_despite_nonzero_actions(self):
        node = method('gym_dcmm/envs/DcmmVecEnv.py', 'DcmmVecEnv', '_step_mujoco_simulation')
        stop = next(i for i, n in enumerate(node.body) if isinstance(n, ast.Expr)
                    and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                    and n.value.func.attr == 'update_target_ctrl')
        node.body = node.body[:stop]
        cfg = SimpleNamespace(arm_joints=np.arange(6), basket_fix_base=False, basket_track_max_speed=0.8)
        robot = SimpleNamespace(target_base_vel=np.zeros(3), target_arm_qpos=np.zeros(6),
                                target_hand_qpos=np.zeros(16))
        env = SimpleNamespace(object_motion='throw_basket', task='Tracking', Dcmm=robot)
        run = execute(node, dict(np=np, DcmmCfg=cfg, limit_speed=PARK.limit_speed))
        run(env, dict(base=np.array([0.2, 0.4]), arm=np.ones(6), hand=np.ones(12)))
        np.testing.assert_allclose(robot.target_base_vel[:2], [0.2, 0.4])
        np.testing.assert_allclose(robot.target_arm_qpos, cfg.arm_joints)
        self.assertEqual(robot.target_hand_qpos[0], 0.6)
        self.assertTrue(env.arm_limit)
        run(env, dict(base=np.array([1.5, 1.5]), arm=np.ones(6), hand=np.ones(12)))
        self.assertAlmostEqual(np.linalg.norm(robot.target_base_vel[:2]), 0.8)

    def test_rotated_parking_target_and_overshoot_direction(self):
        state = PARK.parking_state(np.array([0., 0., 0.4]), [0., 0.8], np.pi / 2,
                                   np.array([0., 2.2, .9]), CFG)
        target_y = 2.2 - CFG.basket_base_front_dist
        np.testing.assert_allclose(state['observation'], [target_y, 0., .5], atol=1e-8)
        np.testing.assert_allclose(state['desired_velocity'], [.8, 0.], atol=1e-8)
        overshot = PARK.parking_state(np.array([0., target_y + .2, .4]), [0., .4], 0.,
                                      np.array([0., 2.2, .9]), CFG)
        self.assertLess(overshot['desired_velocity'][1], 0.)

    def test_reward_encourages_approach_then_braking(self):
        def state(y, speed):
            return PARK.parking_state(np.array([0., y, .4]), [0., speed], 0.,
                                      np.array([0., 2.2, .9]), CFG)
        def reward(s, previous, success=False):
            return PARK.parking_reward(s, previous, np.zeros(2), success, False, CFG)[0]
        target_y = 2.2 - CFG.basket_base_front_dist
        self.assertLess(reward(state(0., 0.), target_y), 0.)  # informative even far from goal
        self.assertGreater(reward(state(.03, .8), target_y), reward(state(0., 0.), target_y))
        self.assertGreater(reward(state(target_y, 0.), .02), reward(state(target_y, .8), .02))
        self.assertGreater(reward(state(target_y, 0.), .02, True), 90.)

    def test_actual_tracking_requires_consecutive_stopped_steps(self):
        node = method('gym_dcmm/envs/DcmmVecEnv.py', 'DcmmVecEnv', 'step')
        basket = next(n for n in node.body if isinstance(n, ast.If)
                      and ast.unparse(n.test) == "self.object_motion == 'throw_basket' and (not self.terminated)")
        code = compile(ast.Module(body=[basket], type_ignores=[]), '<basket termination>', 'exec')
        env = SimpleNamespace(object_motion='throw_basket', task='Tracking', terminated=False,
                              basket_settle_steps=0, _basket_parking_state=lambda: {'settled': True})
        info = {'success': False}
        scope = dict(self=env, info=info, DcmmCfg=CFG)
        for _ in range(4):
            exec(code, scope)
            self.assertFalse(info['success'])
        env._basket_parking_state = lambda: {'settled': False}
        exec(code, scope)
        self.assertEqual(env.basket_settle_steps, 0)
        env._basket_parking_state = lambda: {'settled': True}
        for _ in range(5):
            exec(code, scope)
        self.assertTrue(info['success'])
        self.assertEqual(env.terminated_reason, 'base_in_front')

    def test_actual_reward_bypasses_ball_rewards(self):
        node = method('gym_dcmm/envs/DcmmVecEnv.py', 'DcmmVecEnv', 'compute_reward')
        run = execute(node, dict(np=np, DcmmCfg=CFG, parking_reward=PARK.parking_reward,
                                 limit_speed=PARK.limit_speed))
        state = PARK.parking_state(np.array([0., 0., .4]), [0., 0.], 0.,
                                   np.array([0., 2.2, .9]), CFG)
        env = SimpleNamespace(object_motion='throw_basket', task='Tracking', terminated=False,
                              basket_previous_distance=state['distance'], _basket_parking_state=lambda: state)
        info = {'success': False}
        value = run(env, {}, info, {'base': np.zeros(2)})
        self.assertLess(value, 0.)
        self.assertEqual(value, sum(info['basket_reward_terms'].values()))

    def test_observation_prefix_matches_tracking(self):
        obs = dict(base=dict(v_lin_2d=np.zeros((1, 2))),
                   arm=dict(ee_pos3d=np.zeros((1, 3)), ee_quat=np.zeros((1, 4)),
                            ee_v_lin_3d=np.zeros((1, 3))),
                   object=dict(pos3d=np.zeros((1, 3)), v_lin_3d=np.zeros((1, 3))),
                   hand=np.full((1, 12), 7), basket=dict(rel_pos3d=np.full((1, 3), 9),
                                                        target_rel_pos2d=np.full((1, 2), 11)))
        env = SimpleNamespace(device='cpu', env=SimpleNamespace(call=lambda _: ['Tracking']))
        values = []
        for filename, cls in [('ppo_dcmm_track.py', 'PPO_Track'),
                              ('ppo_dcmm_catch_two_stage.py', 'PPO_Catch_TwoStage')]:
            node = method('gym_dcmm/algs/ppo_dcmm/' + filename, cls, 'obs2tensor')
            values.append(execute(node, dict(np=np, torch=torch))(env, obs))
        torch.testing.assert_close(values[0], values[1][:, :-12])
        self.assertTrue(torch.all(values[1][:, -12:] == 7))

    def test_policy_split_and_tracking_weight_transfer(self):
        modules = []
        for name in ['models_track', 'models_catch']:
            spec = importlib.util.spec_from_file_location(name, ROOT / 'gym_dcmm/algs/ppo_dcmm' / (name + '.py'))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            modules.append(module)
        for tracking_dim, obs_dim in [(2, 35), (8, 30)]:
            common = dict(actor_units=[16, 16], separate_value_mlp=True)
            track = modules[0].ActorCritic(dict(common, actions_num=tracking_dim, input_shape=(obs_dim - 12,)))
            catch = modules[1].ActorCritic(dict(common, actions_num=20, tracking_actions_num=tracking_dim,
                                               input_shape=(obs_dim,)))
            catch.actor_mlp_t.load_state_dict(track.actor_mlp.state_dict())
            catch.mu_t.load_state_dict(track.mu.state_dict())
            self.assertEqual(catch.mu_c.out_features, 20 - tracking_dim)
            x = torch.zeros((2, obs_dim))
            output = catch.act(dict(obs=x, obs_t=x[:, :-12], obs_c=x[:, 2:]))
            self.assertEqual(output['actions'].shape, (2, 20))


if __name__ == '__main__':
    unittest.main()
