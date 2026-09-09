"""CPU checks of actual control code, observation packing and policy transfer."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def method(path, cls, name):
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name)


def execute(node, scope):
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual code>', 'exec'), scope)
    return scope[node.name]


class BasketTests(unittest.TestCase):
    def test_tracking_holds_arm_and_hand_despite_nonzero_actions(self):
        node = method('gym_dcmm/envs/DcmmVecEnv.py', 'DcmmVecEnv', '_step_mujoco_simulation')
        stop = next(i for i, n in enumerate(node.body) if isinstance(n, ast.Expr)
                    and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                    and n.value.func.attr == 'update_target_ctrl')
        node.body = node.body[:stop]
        cfg = SimpleNamespace(arm_joints=np.arange(6), basket_fix_base=False)
        robot = SimpleNamespace(target_base_vel=np.zeros(3), target_arm_qpos=np.zeros(6),
                                target_hand_qpos=np.zeros(16))
        env = SimpleNamespace(object_motion='throw_basket', task='Tracking', Dcmm=robot)
        run = execute(node, dict(np=np, DcmmCfg=cfg))
        run(env, dict(base=np.array([0.2, 0.4]), arm=np.ones(6), hand=np.ones(12)))
        np.testing.assert_allclose(robot.target_base_vel[:2], [0.2, 0.4])
        np.testing.assert_allclose(robot.target_arm_qpos, cfg.arm_joints)
        self.assertEqual(robot.target_hand_qpos[0], 0.6)
        self.assertTrue(env.arm_limit)

    def test_observation_prefix_matches_tracking(self):
        obs = dict(base=dict(v_lin_2d=np.zeros((1, 2))),
                   arm=dict(ee_pos3d=np.zeros((1, 3)), ee_quat=np.zeros((1, 4)),
                            ee_v_lin_3d=np.zeros((1, 3))),
                   object=dict(pos3d=np.zeros((1, 3)), v_lin_3d=np.zeros((1, 3))),
                   hand=np.full((1, 12), 7), basket=dict(rel_pos3d=np.full((1, 3), 9)))
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
        for tracking_dim, obs_dim in [(2, 33), (8, 30)]:
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
