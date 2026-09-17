"""Baseline integrity and routing checks without a MuJoCo runtime."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]

class BaselineTests(unittest.TestCase):
    def test_frozen_sources_and_shared_dependencies(self):
        manifest = json.loads((ROOT / 'BOUNCE_3A36F5F_MANIFEST.json').read_text())
        for item in manifest['files']:
            content = (ROOT / item['destination']).read_bytes().replace(b'\r\n', b'\n')
            self.assertEqual(hashlib.sha256(content).hexdigest(), item['restored_lf_sha256'], item['destination'])
            ast.parse(content)
        for item in manifest['shared_dependencies']:
            content = (ROOT / item['path']).read_bytes()
            if item['mode'] == 'lf':
                content = content.replace(b'\r\n', b'\n')
            elif item['mode'] == 'xml':
                content = ET.canonicalize(content.decode(), strip_text=True).encode()
            self.assertEqual(hashlib.sha256(content).hexdigest(), item['sha256'], item['path'])

    def test_current_constructor_routes_only_bounce_including_positional(self):
        tree = ast.parse((ROOT / 'gym_dcmm/envs/DcmmVecEnv.py').read_text(encoding='utf-8'))
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DcmmVecEnv')
        new = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == '__new__')
        # Keep the real constructor signature, replacing its simulation body.
        init = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        init.body = [ast.Pass()]
        node.bases = []; node.body = [new, init]
        module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
        scope = {}
        stub = ModuleType('gym_dcmm.envs.bounce_compat')
        stub.make_bounce_3a36f5f = lambda parameters: SimpleNamespace(parameters=parameters)
        roll_stub = ModuleType('gym_dcmm.envs.roll_compat')
        roll_stub.make_roll_645edc4 = lambda parameters: SimpleNamespace(parameters=parameters)
        with patch.dict(sys.modules, {'gym_dcmm.envs.bounce_compat': stub, roll_stub.__name__: roll_stub}):
            exec(compile(module, '<actual-constructor>', 'exec'), scope)
            cls = scope['DcmmVecEnv']
            for mode in ['bounce', 'tan', chr(0x5f39)]:
                env = cls(task='Tracking', object_motion=mode)
                self.assertEqual(env.parameters['object_motion'], mode)
            self.assertEqual(cls(task='Tracking', object_motion='roll').parameters['object_motion'], 'roll')
            for mode in ['throw', 'throw_basket']:
                self.assertIsInstance(cls(task='Tracking', object_motion=mode), cls)
            import inspect
            values = [p.default for name, p in inspect.signature(cls.__init__).parameters.items() if name != 'self']
            names = list(inspect.signature(cls.__init__).parameters)[1:]
            values[names.index('object_motion')] = 'bounce'
            self.assertEqual(cls(*values).parameters['object_motion'], 'bounce')
            values[names.index('object_motion')] = 'roll'
            self.assertEqual(cls(*values).parameters['object_motion'], 'roll')

    def test_adapter_rejects_presets_and_preserves_old_arguments(self):
        scope = {}
        exec(compile((ROOT / 'gym_dcmm/envs/bounce_compat.py').read_text(), '<adapter>', 'exec'), scope)
        factory = scope['make_bounce_3a36f5f']
        for key in ['bounce_physics', 'bounce_launch']:
            with self.assertRaises(ValueError):
                factory({key: 'P0' if key.endswith('physics') else 'L0'})
        stub = ModuleType('gym_dcmm.envs.DcmmVecEnv_bounce_3a36f5f')
        stub.DcmmVecEnv = lambda **kwargs: kwargs
        with patch.dict(sys.modules, {stub.__name__: stub}):
            result = factory(dict(task='Catching', object_motion='tan', bounce_physics='legacy',
                                  bounce_launch='legacy', bounce_log=False, roll_log=True, env_time=2.5))
        self.assertEqual(result, dict(task='Catching', object_motion='bounce', env_time=2.5))

    def test_training_selects_frozen_agents_only_for_bounce(self):
        tree = ast.parse((ROOT / 'train_DCMM.py').read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        selection = main.body[0]
        stubs = {}
        for suffix, symbol in [('track','PPO_Track'), ('catch_two_stage','PPO_Catch_TwoStage'), ('catch_one_stage','PPO_Catch_OneStage')]:
            name = 'gym_dcmm.algs.ppo_dcmm_3a36f5f.ppo_dcmm_' + suffix
            stub = ModuleType(name); setattr(stub, symbol, 'old_' + symbol); stubs[name] = stub
            name = name.replace('3a36f5f', '645edc4')
            stub = ModuleType(name); setattr(stub, symbol, 'roll_' + symbol); stubs[name] = stub
        code = compile(ast.Module(body=[selection], type_ignores=[]), '<actual-agent-selection>', 'exec')
        with patch.dict(sys.modules, stubs):
            for mode in ['bounce', 'roll', 'throw_basket']:
                scope = dict(config=SimpleNamespace(object_motion=mode), print=lambda *a, **kw: None,
                             PPO_Track='track', PPO_Catch_TwoStage='two', PPO_Catch_OneStage='one')
                exec(code, scope)
                self.assertEqual(scope['TrackingAgent'], 'old_PPO_Track' if mode == 'bounce' else ('roll_PPO_Track' if mode == 'roll' else 'track'))
                self.assertEqual(scope['TwoStageAgent'], 'old_PPO_Catch_TwoStage' if mode == 'bounce' else ('roll_PPO_Catch_TwoStage' if mode == 'roll' else 'two'))

if __name__ == '__main__':
    unittest.main()
