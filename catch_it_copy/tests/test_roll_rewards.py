"""CPU checks of real roll reward/transition code and table-clearance geometry."""
import ast
import importlib.util
import math
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


CFG = load('roll_cfg', 'configs/env/DcmmCfg.py')
R = load('roll_rewards', 'gym_dcmm/utils/roll_rewards.py')
TREE = ast.parse((ROOT / 'gym_dcmm/envs/DcmmVecEnv.py').read_text(encoding='utf-8'))
CLS = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == 'DcmmVecEnv')


def method(name):
    return next(n for n in CLS.body if isinstance(n, ast.FunctionDef) and n.name == name)


SCOPE = dict(np=np, math=math, DcmmCfg=CFG, interception_target=R.interception_target,
             position_terms=R.position_terms, hand_terms=R.hand_terms,
             hand_workspace=R.hand_workspace, capture_ready=R.capture_ready,
             wait_target=R.wait_target, quaternion_to_rotation_matrix=lambda _: np.eye(3))
for name in ('_roll_interception_state', 'compute_reward'):
    exec(compile(ast.Module(body=[method(name)], type_ignores=[]), '<actual-roll>', 'exec'), SCOPE)

# Execute the actual touch-filter and stage-change blocks together. This exercises
# the gate used before rewards and Tracking completion without importing MuJoCo.
TOUCH_FILTER = next(n for n in method('step').body if isinstance(n, ast.If)
                    and 'roll_raw_touch' in ast.unparse(n))
TRANSITION = next(n for n in method('step').body if isinstance(n, ast.If)
                  and "self.task == 'Catching'" in ast.unparse(n.test)
                  and 'self.stage' in ast.unparse(n))
TRANSITION_CODE = compile(ast.Module(body=[TOUCH_FILTER, TRANSITION], type_ignores=[]),
                          '<actual-roll-transition>', 'exec')


def fixture():
    """A low, clear hand and a ball already outside the front edge."""
    ee = SimpleNamespace(xpos=np.array([0., .65, CFG.roll_wait_height]),
                         cvel=np.zeros(6),
                         xmat=np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]))
    obj = SimpleNamespace(xpos=ee.xpos + [0., 0., CFG.roll_intercept_ball_offset])
    # Geometry 2 is visual-only inside the table. Geometry 3 has conaffinity only.
    geom_xpos = np.array([CFG.roll_table_pos, [0., .65, .26],
                         CFG.roll_table_pos, [0., .69, .29], obj.xpos])
    model = SimpleNamespace(geom_size=np.array([[0., 0., 0.]] * 4 + [[.04, 0., 0.]]),
                            geom_contype=np.array([1, 1, 0, 0, 1]),
                            geom_conaffinity=np.array([1, 0, 0, 1, 1]),
                            geom_rbound=np.array([2., .025, .5, .02, .04]),
                            opt=SimpleNamespace(gravity=np.array([0., 0., -9.81])))
    data = SimpleNamespace(body=lambda name: obj if name == 'object' else ee,
                           geom_xpos=geom_xpos, qpos=np.zeros(44), qvel=np.zeros(42), time=1.)
    env = SimpleNamespace(object_motion='roll', task='Catching', stage='tracking',
                          Dcmm=SimpleNamespace(data=data, model=model),
                          object_name='object', object_id=4, hand_start_id=1,
                          table_geom_id=0, floor_id=9, _prev_roll_d=None,
                          _roll_locked_wait_target=None, terminated=False,
                          terminated_reason=None, step_touch=True, reward_touch=0.,
                          start_time=0., info=dict(base_distance=0., ee_distance=.04),
                          contacts=dict(base_contacts=np.array([]), object_contacts=np.array([1])),
                          arm_limit=True, norm_ctrl=lambda *_: 0., print_reward=False,
                          roll_palm_face_cos=CFG.roll_palm_face_cos,
                          roll_tracking_xy_thresh=CFG.roll_tracking_xy_thresh,
                          roll_tracking_z_thresh=CFG.roll_tracking_z_thresh)
    env._roll_interception_state = MethodType(SCOPE['_roll_interception_state'], env)
    # Real policy observation is a 12-vector, never a 16-joint dictionary.
    obs = dict(hand=np.zeros(12), arm=dict(ee_pos3d=ee.xpos.copy(),
                                         ee_quat=np.array([1., 0., 0., 0.])),
               object=dict(pos3d=obj.xpos.copy(), v_lin_3d=np.zeros(3)))
    return env, obs


def set_closed_hand(env):
    joints = env.Dcmm.data.qpos[21:37]
    joints[[0, 2, 3, 4, 6, 7, 8, 10, 11]] = np.tile(CFG.roll_hand_close_target, 3)
    joints[13:16] = CFG.roll_hand_thumb_target


def reward(env, obs, ee_distance=.04):
    env._prev_roll_d = None
    info = dict(base_distance=0., ee_distance=ee_distance, env_time=1.)
    value = SCOPE['compute_reward'](env, obs, info, {})
    return value, info


class RollTests(unittest.TestCase):
    def test_predicts_below_table_and_does_not_chase_high_ball_after_exit(self):
        target, waiting = R.interception_target([0., 2., .5], [.2, -1., 0.], .04, CFG)
        self.assertTrue(waiting)
        self.assertLess(target[1], .9)
        self.assertLess(target[2], .42)
        self.assertGreater(target[0], .2)
        target, waiting = R.interception_target([0., .85, .49], [0., -1., -.1], .04, CFG)
        self.assertFalse(waiting)
        self.assertAlmostEqual(target[2], CFG.roll_wait_height)
        target, _ = R.interception_target([0., .5, .2], [0., -1., -1.], .04, CFG)
        self.assertLess(target[2], CFG.roll_wait_height)

    def test_slow_ball_target_stays_on_ballistic_path_without_horizontal_clamp(self):
        edge_y = CFG.roll_table_pos[1] - CFG.roll_table_size[1]
        target, _ = R.interception_target([0., 1.1, .5], [0., -.1, 0.], .04, CFG)
        fall_time = np.sqrt(2. * (.5 - CFG.roll_wait_height - CFG.roll_intercept_ball_offset) / 9.81)
        self.assertAlmostEqual(target[1], edge_y - .1 * fall_time)
        self.assertGreater(target[1], edge_y - .12)

    def test_hand_workspace_checks_fingertips_and_all_three_table_dimensions(self):
        ee = np.array([0., .65, .24])
        clear = R.hand_workspace([[0., .65, .24]], [.02], ee, CFG)
        self.assertTrue(clear['hand_clear'])
        dangerous = R.hand_workspace([[0., .65, .24], [0., .87, .44]], [.02, .025], ee, CFG)
        self.assertAlmostEqual(dangerous['clearance'], .005)
        self.assertFalse(dangerous['hand_clear'])
        # Below the slab is safe despite overlapping the table's XY footprint.
        below = R.hand_workspace([[0., 1., .30]], [.02], ee, CFG)
        self.assertAlmostEqual(below['clearance'], .10)
        self.assertTrue(below['hand_clear'])
        inside = R.hand_workspace([CFG.roll_table_pos], [.02], ee, CFG)
        self.assertLess(inside['clearance'], 0.)
        with self.assertRaises(ValueError):
            R.hand_workspace([], [], ee, CFG)

    def test_workspace_lowers_wait_height_for_tall_hand_and_reports_infeasible(self):
        ee = np.array([0., .65, .24])
        tall = R.hand_workspace([[0., .65, .40]], [.03], ee, CFG)
        self.assertLess(tall['wait_height'], CFG.roll_wait_height)
        self.assertTrue(tall['height_feasible'])
        too_tall = R.hand_workspace([[0., .65, .70]], [.03], ee, CFG)
        self.assertEqual(too_tall['wait_height'], CFG.roll_min_wait_height)
        self.assertFalse(too_tall['height_feasible'])

    def test_actual_state_ignores_visual_geoms_but_includes_conaffinity_only_finger(self):
        env, _ = fixture()
        self.assertTrue(env._roll_interception_state()['hand_clear'])
        env.Dcmm.data.geom_xpos[3] = [0., .87, .44]
        state = env._roll_interception_state()
        self.assertAlmostEqual(state['clearance'], .01)
        self.assertFalse(state['hand_clear'])

    def test_below_table_waiting_and_braking_outscore_climbing(self):
        target = np.array([0., .65, CFG.roll_wait_height])
        def position_reward(pos, velocity):
            return sum(R.position_terms(pos, velocity, target, True, None, CFG)[0].values())
        self.assertGreater(position_reward(target, np.zeros(3)),
                           position_reward([0., .65, .5], np.zeros(3)))
        self.assertGreater(position_reward(target, np.zeros(3)),
                           position_reward(target, [0., 0., .5]))

    def test_finger_shaping_is_bounded_coordinated_and_gated(self):
        env, _ = fixture()
        set_closed_hand(env)
        hand = env.Dcmm.data.qpos[21:37].copy()
        good = R.hand_terms(hand, .05, True, CFG)
        self.assertGreater(good['hand_closure'], 0.)
        self.assertEqual(R.hand_terms(hand, 1., False, CFG)['hand_closure'], 0.)
        self.assertEqual(R.hand_terms(hand, .01, True, CFG, allow_closure=False)['hand_closure'], 0.)
        bad = hand.copy()
        bad[0:4] = [2.2, 0., -1., 2.2]
        self.assertGreater(sum(good.values()), sum(R.hand_terms(bad, .05, True, CFG).values()))
        with self.assertRaises(ValueError):
            R.hand_terms(np.zeros(12), .05, True, CFG)

    def test_actual_reward_uses_height_and_real_16_joint_state_in_both_stages(self):
        for stage in ('tracking', 'grasping'):
            env, obs = fixture()
            env.stage = stage
            open_reward, _ = reward(env, obs)
            set_closed_hand(env)
            closed_reward, info = reward(env, obs)
            self.assertGreater(closed_reward, open_reward)
            self.assertGreater(info['roll_reward_terms']['height'], 0.)
            self.assertGreater(info['roll_reward_terms']['hand_closure'], 0.)

    def test_contact_only_counts_after_whole_ball_and_hand_clear_table(self):
        cases = ('on_table', 'ball_overlaps_edge', 'still_table_contact', 'finger_too_close',
                 'clear', 'table_collision')
        for case in cases:
            with self.subTest(case=case):
                env, obs = fixture()
                if case == 'on_table':
                    env.Dcmm.data.body('object').xpos[:] = [0., 1., .5]
                    env.contacts['object_contacts'] = np.array([0, 1])
                elif case == 'ball_overlaps_edge':
                    env.Dcmm.data.body('object').xpos[:] = [0., .88, .45]
                elif case == 'still_table_contact':
                    env.contacts['object_contacts'] = np.array([0, 1])
                elif case == 'finger_too_close':
                    env.Dcmm.data.geom_xpos[3] = [0., .87, .44]
                elif case == 'table_collision':
                    env.terminated = True
                    env.terminated_reason = 'table_collision'
                info = {}
                exec(TRANSITION_CODE, dict(SCOPE, self=env, obs=obs, info=info))
                self.assertTrue(info['roll_raw_touch'])
                self.assertEqual(env.stage, 'grasping' if case == 'clear' else 'tracking')
                if case not in ('clear', 'table_collision'):
                    self.assertFalse(env.step_touch)
                    set_closed_hand(env)
                    _, evaluated = reward(env, obs)
                    self.assertEqual(evaluated['roll_reward_terms']['hand_closure'], 0.)
                    self.assertEqual(env.reward_touch, 0.)

    def test_locked_target_does_not_follow_ball_or_jump_on_grasping(self):
        env, obs = fixture()
        obj = env.Dcmm.data.body('object')
        obj.xpos[:] = [0., 1.5, .5]
        env.Dcmm.data.qvel[36:39] = [0., -1., 0.]
        first = env._roll_interception_state()['target']
        self.assertIsNone(env._roll_locked_wait_target)
        env.Dcmm.data.body('link6').xpos[:] = first
        env.Dcmm.data.geom_xpos[1] = first + [0., 0., .02]
        env.Dcmm.data.geom_xpos[3] = first + [0., .04, .05]
        locked = env._roll_interception_state()
        self.assertTrue(locked['target_locked'])
        obj.xpos[:] = [0., .84, .43]
        np.testing.assert_allclose(env._roll_interception_state()['target'], first)
        _, tracking_info = reward(env, obs)
        env.stage = 'grasping'
        _, grasping_info = reward(env, obs)
        np.testing.assert_allclose(grasping_info['roll_target_world'], first)
        np.testing.assert_allclose(grasping_info['roll_target_world'], tracking_info['roll_target_world'])
        grasping_info['roll_target_world'][1] = -99.
        np.testing.assert_allclose(env._roll_locked_wait_target, first)

    def test_actual_grasp_reward_has_no_generic_wrist_ball_precision_bonus(self):
        env, obs = fixture()
        env.stage = 'grasping'
        far, _ = reward(env, obs, ee_distance=.7)
        near, _ = reward(env, obs, ee_distance=.001)
        self.assertAlmostEqual(far, near)
        env.Dcmm.data.geom_xpos[3] = [0., .87, .44]
        unsafe, info = reward(env, obs, ee_distance=.001)
        self.assertLess(info['roll_reward_terms']['workspace'], 0.)
        self.assertEqual(info['roll_reward_terms']['hand_closure'], 0.)
        self.assertLess(unsafe, near)


if __name__ == '__main__':
    unittest.main()
