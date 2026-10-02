import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
source = ast.parse((ROOT/'gym_dcmm/agents/MujocoDcmm.py').read_text(encoding='utf-8-sig'))
method = next(n for n in ast.walk(source) if isinstance(n, ast.FunctionDef) and n.name == 'move_ee_pose')
scope = dict(np=np, R=R, DEBUG_ARM=False, mujoco=SimpleNamespace(mj_fwdPosition=lambda *a: None))
exec(compile(ast.Module(body=[method], type_ignores=[]), '<actual-method>', 'exec'), scope)
spec = importlib.util.spec_from_file_location('probe', ROOT/'basket_throw_probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)

class BasketControlTests(unittest.TestCase):
    def fixture(self, ok=True):
        q = np.array([.7, .2, -.3, .6]); q /= np.linalg.norm(q)
        body = SimpleNamespace(xpos=np.zeros(3), xquat=q)
        captured = {}
        def solve(p, quat):
            captured['quat'] = quat.copy()
            captured['seed'] = robot.data_arm.qpos.copy()
            return np.ones(6)*2, ok
        robot = SimpleNamespace(data=SimpleNamespace(qpos=np.arange(40.)/100),
            data_arm=SimpleNamespace(qpos=np.zeros(6), body=lambda n: body),
            model_arm=None, current_ee_pos=np.zeros(3), current_ee_quat=np.zeros(4),
            ik_arm_solve=solve)
        return robot, q, captured

    def test_zero_rotation_keeps_wxyz_and_seeds_measured_joints(self):
        robot, q, capture = self.fixture()
        scope['move_ee_pose'](robot, np.zeros(6), measured_state=True)
        np.testing.assert_allclose(capture['quat'], q)
        np.testing.assert_allclose(capture['seed'], robot.data.qpos[15:21])

    def test_rotation_composition_matches_wxyz_contract(self):
        robot, q, capture = self.fixture()
        delta = np.array([0., 0., 0., .1, -.2, .3])
        scope['move_ee_pose'](robot, delta, measured_state=True)
        actual = R.from_quat(capture['quat'][[1,2,3,0]]).as_matrix()
        expected = (R.from_euler('zxy', delta[3:])*R.from_quat(q[[1,2,3,0]])).as_matrix()
        np.testing.assert_allclose(actual, expected)

    def test_failed_solution_does_not_poison_internal_seed(self):
        robot, _, _ = self.fixture(False)
        scope['move_ee_pose'](robot, np.zeros(6), measured_state=True)
        np.testing.assert_allclose(robot.data_arm.qpos, robot.data.qpos[15:21])

    def test_other_modes_keep_legacy_default(self):
        robot, q, capture = self.fixture()
        scope['move_ee_pose'](robot, np.zeros(6))
        np.testing.assert_allclose(capture['seed'], np.zeros(6))
        np.testing.assert_allclose(robot.data_arm.qpos, np.ones(6)*2)

    def test_probe_has_hold_back_swing_and_clamped_opening(self):
        back, forward = np.array([-.1]), np.array([.2])
        for t, target in [(0., 0.), (.5, 0.), (1., -.1), (1.5, .2), (2., .2)]:
            np.testing.assert_allclose(probe.trajectory(t,.5,.5,.5,back,forward), [target])
        self.assertEqual(probe.smooth_segment(-1., .1), 0.)
        self.assertEqual(probe.smooth_segment(2., .1), 1.)

if __name__ == '__main__':
    unittest.main()
