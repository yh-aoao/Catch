import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TREE = ast.parse((ROOT / 'gym_dcmm/envs/DcmmVecEnv_bounce_61709fc.py').read_text(encoding='utf-8'))
CLASS = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == 'DcmmVecEnv')

class HeadlessTests(unittest.TestCase):
    def test_renderer_creation_is_opt_in(self):
        init = next(n for n in CLASS.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        nodes = []
        for n in init.body:
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == 'mujoco_renderer' for t in n.targets):
                nodes.append(n)
            if isinstance(n, ast.If) and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == 'MujocoRenderer' for c in ast.walk(n)):
                nodes.append(n)
        self.assertEqual(len(nodes), 2)
        code = compile(ast.Module(body=nodes, type_ignores=[]), '<renderer-init>', 'exec')
        for camera, viewer, per_step in [(False,False,False),(True,False,False),(False,True,False),(False,False,True)]:
            env = SimpleNamespace(imshow_cam=camera, render_per_step=per_step, Dcmm=SimpleNamespace(open_viewer=viewer, model=1, data=2))
            calls = []
            def renderer(*args):
                calls.append(args)
                return 'renderer'
            exec(code, dict(self=env, MujocoRenderer=renderer))
            self.assertEqual(len(calls), int(camera or viewer or per_step))
            self.assertEqual(env.mujoco_renderer, 'renderer' if calls else None)

    def test_render_without_gl_and_without_random_draws(self):
        method = next(n for n in CLASS.body if isinstance(n, ast.FunctionDef) and n.name == 'render')
        scope = dict(np=np)
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<render>', 'exec'), scope)
        state = np.random.get_state()
        env = SimpleNamespace(mujoco_renderer=None, img_size=(480,640))
        images = scope['render'](env)
        self.assertEqual(images.shape, (0,480,640))
        after = np.random.get_state()
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])

if __name__ == '__main__':
    unittest.main()
