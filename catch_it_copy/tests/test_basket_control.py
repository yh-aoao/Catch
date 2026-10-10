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
            return np.ones(6)*2, ok, 4, .001, True, .002
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

    def test_zero_back_time_skips_nonzero_back_offset_without_jump(self):
        back, forward = np.array([-.1, -.02]), np.array([.16, .12])
        np.testing.assert_allclose(probe.trajectory(.5,.5,0.,.4,back,forward), [0.,0.])
        np.testing.assert_allclose(probe.trajectory(.7,.5,0.,.4,back,forward), forward*.5)
        np.testing.assert_allclose(probe.trajectory(.9,.5,0.,.4,back,forward), forward)

    def test_scan_covers_cells_and_repeats_each(self):
        args = SimpleNamespace(scan=True, scan_swing_times=[.3,.6],
            scan_release_fractions=[.3,.5,.7], episodes=2)
        trials = probe.trial_settings(args)
        self.assertEqual(len(trials), 12)
        self.assertEqual(len(set(trials)), 6)
        self.assertTrue(all(trials.count(pair) == 2 for pair in set(trials)))

    def test_release_diagnostic_does_not_call_downward_drop_a_throw(self):
        snapshot = dict(last_contact=dict(time=10.73, velocity=[0.,.5,-1.2]))
        result = probe.release_diagnostic(snapshot, 10., 1., 1.315, np.array([0.,1.]))
        self.assertEqual(result['status'], 'lost_before_forward_swing')
        self.assertFalse(result['upward_forward_flight'])
        snapshot['last_contact'] = dict(time=11.4, velocity=[0.,2.,3.])
        result = probe.release_diagnostic(snapshot, 10., 1., 1.315, np.array([0.,1.]))
        self.assertTrue(result['upward_forward_flight'])
        self.assertEqual(result['status'], 'released_after_planned_open')
        self.assertEqual(probe.release_diagnostic(None, 0., .5, .7, np.array([0.,1.]))['status'],
                         'no_free_flight_detected')

    def test_joint_direction_report_restores_internal_kinematics(self):
        robot, _, _ = self.fixture()
        robot.model = SimpleNamespace(jnt_limited=np.ones(6, dtype=bool),
                                      jnt_range=np.tile([-1.,1.], (6,1)))
        body = SimpleNamespace(xpos=np.zeros(3), xmat=np.eye(3).reshape(-1))
        robot.data_arm.body = lambda n: body
        saved = robot.data_arm.qpos.copy()
        def forward(model, data):
            body.xpos[:] = data.qpos[:3]
        rows = probe.joint_direction_report(robot, SimpleNamespace(mj_fwdPosition=forward), range(6), .1)
        self.assertEqual(len(rows), 12)
        np.testing.assert_allclose(rows[0]['ee_delta_arm_frame'], [-.1,0.,0.])
        np.testing.assert_allclose(robot.data_arm.qpos, saved)

    def test_absolute_probe_goal_does_not_follow_measured_sag(self):
        robot, q, capture = self.fixture()
        robot.model = SimpleNamespace(jnt_limited=np.ones(6, dtype=bool),
                                      jnt_range=np.tile([-3.,3.], (6,1)))
        # Supply a nontrivial fixed pose while actual joints change between calls.
        target = np.array([.1,.3,.5])
        captured_positions = []
        def solve(position, quaternion):
            captured_positions.append(position.copy())
            np.testing.assert_allclose(quaternion, q)
            return np.ones(6)*.2, True, 4, .001, True, .002
        robot.ik_arm_solve = solve
        for drift in [0., -.15]:
            robot.data.qpos[15:21] += drift
            result, ok = probe.solve_absolute_probe_pose(robot, scope['mujoco'], target, q, np.zeros(6), range(6))
            self.assertTrue(ok)
            np.testing.assert_allclose(robot.data_arm.qpos, robot.data.qpos[15:21])
        np.testing.assert_allclose(captured_positions, [target,target])

    def test_absolute_probe_failure_retains_previous_joint_command(self):
        robot, q, _ = self.fixture(False)
        robot.model = SimpleNamespace(jnt_limited=np.ones(6, dtype=bool),
                                      jnt_range=np.tile([-1.,1.], (6,1)))
        previous = np.ones(6)*.3
        for solution, success in [(np.ones(6),False), (np.full(6,np.nan),True), (np.ones(6)*2,True)]:
            robot.ik_arm_solve = lambda p, quat, s=solution, ok=success: (s, ok, 4, .001, True, .002)
            result, ok = probe.solve_absolute_probe_pose(robot, scope['mujoco'], np.zeros(3), q, previous, range(6))
            self.assertFalse(ok)
            np.testing.assert_allclose(result, previous)

if __name__ == '__main__':
    unittest.main()
