import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def tree(path):
    return ast.parse((ROOT / path).read_text(encoding='utf-8'))


TRACK = tree('gym_dcmm/algs/ppo_dcmm/ppo_dcmm_track.py')
SCOPE = dict(np=np, torch=torch)
nodes = [n for n in TRACK.body if isinstance(n, ast.FunctionDef) and n.name == 'terminal_metrics']
exec(compile(ast.Module(body=nodes, type_ignores=[]), '<metrics>', 'exec'), SCOPE)


class MetricsTests(unittest.TestCase):
    def test_contact_timeout_and_failure_priority(self):
        env_tree = tree('gym_dcmm/envs/DcmmVecEnv.py')
        cls = next(n for n in env_tree.body if isinstance(n, ast.ClassDef) and n.name == 'DcmmVecEnv')
        step = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'step')
        node = next(n for n in step.body if isinstance(n, ast.If)
                    and any(isinstance(a, ast.Assign) and any(isinstance(t, ast.Subscript)
                            and isinstance(t.value, ast.Name) and t.value.id == 'info'
                            for t in a.targets) for a in n.body)
                    and 'Tracking' in ast.unparse(n.test))
        code = compile(ast.Module(body=[node], type_ignores=[]), '<success>', 'exec')
        for touch, failed, done, expected, reason in [
            (True, False, True, True, 'track_success'),
            (False, False, True, False, 'timeout'),
            (True, True, True, False, 'out_of_bounds'),
            (False, False, False, False, None),
        ]:
            env = SimpleNamespace(object_motion='bounce', task='Tracking', step_touch=touch,
                                  terminated_reason='out_of_bounds' if failed else None)
            info = {}
            exec(code, dict(self=env, info=info, terminated=failed, done=done))
            self.assertEqual(info['success'], expected)
            self.assertEqual(info.get('terminated_reason'), reason)

    def test_autoreset_terminal_info_overrides_reset_info(self):
        info = dict(success=np.array([False, False]),
                    final_info=np.array([dict(success=True, terminated_reason='track_success'), None], dtype=object),
                    _final_info=np.array([True, False]))
        success, reasons = SCOPE['terminal_metrics'](info, np.array([True, False]), np.array([True, False]))
        self.assertEqual(success.tolist(), [True, False])
        self.assertEqual(reasons[0], 'track_success')

    def test_actual_test_loop_budget_and_cumulative_counts(self):
        cls = next(n for n in TRACK.body if isinstance(n, ast.ClassDef) and n.name == 'PPO_Track')
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('test', 'play_test_steps')]
        scope = dict(SCOPE)
        exec(compile(ast.Module(body=methods, type_ignores=[]), '<test loop>', 'exec'), scope)
        calls = []
        def step(action):
            calls.append(1)
            success = len(calls) % 2 == 1
            return {}, np.array([1.]), np.array([False]), np.array([True]), {
                'success': np.array([False]),
                'final_info': [dict(success=success, terminated_reason='track_success' if success else 'timeout')],
            }
        meter = SimpleNamespace(update=lambda x: None)
        agent = SimpleNamespace(
            env=SimpleNamespace(reset=lambda: ({}, {}), step=step), set_eval=lambda: None,
            obs2tensor=lambda obs: torch.zeros((1, 18)),
            model_act=lambda obs, inference: {'actions': torch.zeros((1, 8))},
            action2dict=lambda action: {}, full_action_dim=20, device='cpu',
            horizon_length=4, num_actors=1, max_test_steps=6, extra_info={},
            current_rewards=torch.zeros((1, 1)), current_lengths=torch.zeros(1),
            episode_test_rewards=meter, episode_test_lengths=meter, episode_test_success=meter,
            storage=SimpleNamespace(data_dict=None))
        agent.play_test_steps = lambda: scope['play_test_steps'](agent)
        scope['test'](agent)
        self.assertEqual(len(calls), 6)
        self.assertEqual(agent.test_completed, 6)
        self.assertEqual(agent.test_successes, 3)
        self.assertEqual(agent.test_reasons, dict(track_success=3, timeout=3))


if __name__ == '__main__':
    unittest.main()
