"""Exercise the real render method without importing MuJoCo/OpenGL."""
import ast
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from distillation_track.teachers import ROOT
from distillation_track.run import env_config


class HeadlessTests(unittest.TestCase):
    def test_bounce_state_only_render_and_viewer(self):
        config = json.loads((ROOT / 'distillation_track/configs/three_tasks.json').read_text())
        self.assertIsNone(env_config(config, 'bounce')['env_kwargs']['render_mode'])
        path = ROOT / 'gym_dcmm/envs/DcmmVecEnv_bounce_july22.py'
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DcmmVecEnv')
        render = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'render')
        ns = {'np': np}
        exec(compile(ast.Module(body=[render], type_ignores=[]), str(path), 'exec'), ns)
        env = SimpleNamespace(render_mode=None, img_size=(112, 112),
                              Dcmm=SimpleNamespace(viewer=None))
        self.assertEqual(ns['render'](env).shape, (0, 112, 112))
        env.Dcmm.viewer = Mock()
        ns['render'](env)
        env.Dcmm.viewer.sync.assert_called_once()
        # Construction must not instantiate a renderer in state-only mode.
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        assignment = next(n for n in ast.walk(init) if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Attribute) and t.attr == 'mujoco_renderer'
                                  for t in n.targets))
        env.Dcmm.model, env.Dcmm.data = object(), object()
        factory = Mock()
        exec(compile(ast.fix_missing_locations(ast.Module(body=[assignment], type_ignores=[])),
                     str(path), 'exec'), {'self': env, 'MujocoRenderer': factory})
        factory.assert_not_called()
        self.assertIsNone(env.mujoco_renderer)
