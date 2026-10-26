"""Check real OneStage action mapping without importing MuJoCo or GPU libraries."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TREE = ast.parse((ROOT / 'gym_dcmm/algs/ppo_dcmm/ppo_dcmm_catch_one_stage.py').read_text(encoding='utf-8'))
SCOPE = dict(np=np)
for name in ('action2dict', '_load_compatible_model_state'):
    method = next(n for n in ast.walk(TREE) if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<actual-onestage>', 'exec'), SCOPE)


class ArrayTensor:
    def __init__(self, data):
        self.data = data
    def cpu(self):
        return self
    def numpy(self):
        return self.data


class FixedBasketTests(unittest.TestCase):
    def test_padded_arm_hand_actions_do_not_shift_into_base(self):
        policy = SimpleNamespace(fixed_basket=True, action_catch_denorm=[1.5, .025, .15])
        actions = np.arange(40, dtype=float).reshape(2, 20)
        result = SCOPE['action2dict'](policy, ArrayTensor(actions))
        np.testing.assert_array_equal(result['base'], np.zeros((2, 2)))
        np.testing.assert_allclose(result['arm'], actions[:, :6] * .025)
        np.testing.assert_allclose(result['hand'], actions[:, 6:18] * .15)

    def test_standard_action_mapping_unchanged(self):
        policy = SimpleNamespace(fixed_basket=False, action_catch_denorm=[1.5, .025, .15],
                                 env=SimpleNamespace(call=lambda name: ['Catching']))
        actions = np.ones((2, 20))
        result = SCOPE['action2dict'](policy, ArrayTensor(actions))
        self.assertEqual(result['hand'].shape, (2, 12))
        np.testing.assert_allclose(result['base'], 1.5)

    def test_baseline_requires_strict_checkpoint_loading(self):
        calls = []
        policy = SimpleNamespace(fixed_basket=True,
            model=SimpleNamespace(load_state_dict=lambda state, strict: calls.append((state, strict))))
        SCOPE['_load_compatible_model_state'](policy, {'weights': 123})
        self.assertEqual(calls, [({'weights': 123}, True)])


if __name__ == '__main__':
    unittest.main()
