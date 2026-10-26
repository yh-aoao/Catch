"""Exercise actual worker NumPy seed expressions without spawning MuJoCo."""
import ast
from pathlib import Path
import unittest
import numpy as np


class WorkerSeedTests(unittest.TestCase):
    def test_initial_and_episode_seeds(self):
        root = Path(__file__).resolve().parents[1]
        for directory in ('distillation_visual', 'distillation_visual_aux'):
            tree = ast.parse((root / directory / 'environments.py').read_text(encoding='utf-8'))
            expressions = [n.args[0] for n in ast.walk(tree) if isinstance(n, ast.Call)
                           and ast.unparse(n.func) == 'np.random.seed']
            self.assertEqual(len(expressions), 2)
            for expression in expressions:
                code = compile(ast.Expression(expression), '<worker-seed>', 'eval')
                for seed, episode in ((123, 0), (123, 10), (5000000123, 0),
                                      (10000020123, 100000), (2**32-1, 1)):
                    value = eval(code, {}, dict(seed=seed, episode=episode))
                    self.assertGreaterEqual(value, 0)
                    self.assertLess(value, 2**32)
                    # A real NumPy initialization, including repeatability.
                    a = np.random.RandomState(value).rand(4)
                    b = np.random.RandomState(value).rand(4)
                    np.testing.assert_array_equal(a, b)
                    original = seed + (episode*100003 if 'episode' in ast.unparse(expression) else 0)
                    if original < 2**32:
                        self.assertEqual(value, original)


if __name__ == '__main__':
    unittest.main()
