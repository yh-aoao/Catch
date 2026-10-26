import ast
import hashlib
import json
from pathlib import Path
from types import ModuleType
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]

class RollBaselineTests(unittest.TestCase):
    def test_frozen_sources(self):
        manifest = json.loads((ROOT / 'ROLL_645EDC4_MANIFEST.json').read_text())
        for item in manifest['files']:
            content = (ROOT / item['destination']).read_bytes().replace(b'\r\n', b'\n')
            self.assertEqual(hashlib.sha256(content).hexdigest(), item['restored_lf_sha256'], item['destination'])
            ast.parse(content)

    def test_adapter_preserves_parameters_and_removes_new_log_flag(self):
        scope = {}
        exec(compile((ROOT / 'gym_dcmm/envs/roll_compat.py').read_text(), '<adapter>', 'exec'), scope)
        stub = ModuleType('gym_dcmm.envs.DcmmVecEnv_roll_645edc4')
        stub.DcmmVecEnv = lambda **kwargs: kwargs
        with patch.dict(sys.modules, {stub.__name__: stub}):
            params = dict(task='Tracking', object_motion='roll', env_time=2.5, roll_log=False, bounce_physics='legacy')
            result = scope['make_roll_645edc4'](params)
        self.assertIn('roll_log', params)
        self.assertEqual(result, dict(task='Tracking', object_motion='roll', env_time=2.5, bounce_physics='legacy'))

if __name__ == '__main__':
    unittest.main()
