"""Structural equivalence and routing checks without a MuJoCo installation."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
MAIN = ast.parse((ROOT / 'gym_dcmm/envs/DcmmVecEnv.py').read_text(encoding='utf-8'))


class Rename(ast.NodeTransformer):
    def __init__(self, names):
        self.names = names
    def visit_Name(self, node):
        node.id = self.names.get(node.id, node.id)
        return node
    def visit_ClassDef(self, node):
        node.name = self.names.get(node.name, node.name)
        return self.generic_visit(node)


def normalize_docstrings(node):
    # Migration removes trailing whitespace, including explanatory docstrings.
    # Do not normalize executable string literals.
    for item in ast.walk(node):
        if isinstance(item, (ast.ClassDef, ast.FunctionDef)) and item.body:
            first = item.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                first.value.value = '\n'.join(line.rstrip() for line in first.value.value.split('\n'))
    return node


class MigrationTests(unittest.TestCase):
    def test_all_historical_methods_are_identical(self):
        cases = [
            ('RollTrackingEnv', 'DcmmVecEnv_roll_track_07c176f.py',
             dict(RollTrackCfg='DcmmCfg', RollTrackRobot='MJ_DCMM',
                  historical_roll_target='roll_wait_target', historical_roll_shaping='roll_wait_shaping',
                  historical_hand_collision_ids='hand_collision_ids')),
            ('BounceEnv', 'DcmmVecEnv_bounce_5fe75d5f.py',
             dict(BounceCfg='DcmmCfg', BounceRobot='MJ_DCMM'))]
        for name, file, names in cases:
            with self.subTest(environment=name):
                actual = copy.deepcopy(next(n for n in MAIN.body if isinstance(n, ast.ClassDef) and n.name == name))
                names[name] = 'DcmmVecEnv'
                actual = Rename(names).visit(actual)
                old = ast.parse((ROOT / 'gym_dcmm/envs' / file).read_text(encoding='utf-8'))
                expected = next(n for n in old.body if isinstance(n, ast.ClassDef) and n.name == 'DcmmVecEnv')
                self.assertEqual(ast.dump(normalize_docstrings(actual)), ast.dump(normalize_docstrings(expected)))

    def factories(self):
        scope = dict(RollTrackingEnv=lambda **kw: SimpleNamespace(kind='roll', parameters=kw),
                     BounceEnv=lambda **kw: SimpleNamespace(kind='bounce', parameters=kw))
        for name in ('_make_main_roll_track', '_make_main_bounce'):
            node = next(n for n in MAIN.body if isinstance(n, ast.FunctionDef) and n.name == name)
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<main-factory>', 'exec'), scope)
        return scope

    def test_factories_preserve_settings_and_validate_presets(self):
        scope = self.factories()
        settings = dict(task='Tracking', object_motion='roll', roll_log=False, env_time=2.5)
        result = scope['_make_main_roll_track'](settings)
        self.assertEqual(result.parameters, dict(task='Tracking', object_motion='roll', env_time=2.5))
        self.assertIn('roll_log', settings)
        result = scope['_make_main_bounce'](dict(task='Catching', object_motion='tan',
            bounce_physics='legacy', bounce_launch='legacy', basket_log=False, roll_log=False))
        self.assertEqual(result.parameters, dict(task='Catching', object_motion='bounce'))
        with self.assertRaises(ValueError):
            scope['_make_main_bounce'](dict(bounce_physics='P0'))

    def test_standard_entry_routes_to_main_factories(self):
        scope = self.factories()
        cls = copy.deepcopy(next(n for n in MAIN.body if isinstance(n, ast.ClassDef) and n.name == 'DcmmVecEnv'))
        cls.bases = []
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('__new__', '__init__')]
        for node in cls.body:
            if node.name == '__init__':
                node.body = [ast.Pass()]
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), '<main-route>', 'exec'), scope)
        for alias in ('bounce', 'tan', '弹'):
            self.assertEqual(scope['DcmmVecEnv'](task='Tracking', object_motion=alias).kind, 'bounce')
        # Roll routing imports the Catch compatibility module; avoid loading gym.
        import sys
        from types import ModuleType
        from unittest.mock import patch
        stub = ModuleType('gym_dcmm.envs.roll_compat')
        stub.make_roll_645edc4 = lambda kw: SimpleNamespace(kind='current_catch')
        with patch.dict(sys.modules, {stub.__name__: stub}):
            self.assertEqual(scope['DcmmVecEnv'](task='Tracking', object_motion='roll').kind, 'roll')
            self.assertEqual(scope['DcmmVecEnv'](task='Catching', object_motion='roll').kind, 'current_catch')
        self.assertIsInstance(scope['DcmmVecEnv'](task='Catching', object_motion='throw_basket'), scope['DcmmVecEnv'])


if __name__ == '__main__':
    unittest.main()
