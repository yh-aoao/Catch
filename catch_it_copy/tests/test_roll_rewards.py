"""CPU regression checks: actual environment rewards, falling targets and hand shaping."""
import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
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


class RollTests(unittest.TestCase):
    def test_predicts_below_table_and_does_not_chase_high_ball_after_exit(self):
        target, waiting = R.interception_target([0., 2., .5], [.2, -1., 0.], .04, CFG)
        self.assertTrue(waiting)
        self.assertLess(target[1], .9 - CFG.roll_table_clearance)
        self.assertLess(target[2], .42)
        self.assertGreater(target[0], .2)
        target, waiting = R.interception_target([0., .85, .49], [0., -1., -.1], .04, CFG)
        self.assertFalse(waiting)
        self.assertAlmostEqual(target[2], CFG.roll_wait_height)
        target, _ = R.interception_target([0., .5, .2], [0., -1., -1.], .04, CFG)
        self.assertLess(target[2], CFG.roll_wait_height)

    def test_below_table_waiting_and_braking_outscore_climbing(self):
        target = np.array([0., .65, CFG.roll_wait_height])
        def reward(pos, velocity):
            return sum(R.position_terms(pos, velocity, target, True, None, CFG)[0].values())
        self.assertGreater(reward(target, np.zeros(3)), reward([0., .65, .5], np.zeros(3)))
        self.assertGreater(reward(target, np.zeros(3)), reward(target, [0., 0., .5]))
        self.assertGreater(reward(target, np.zeros(3)), reward([0., .95, .3], np.zeros(3)))

    def test_finger_shaping_is_bounded_coordinated_and_gated(self):
        hand = np.zeros(16)
        hand[[0, 2, 3, 4, 6, 7, 8, 10, 11]] = np.tile(CFG.roll_hand_close_target, 3)
        hand[13:16] = CFG.roll_hand_thumb_target
        good = R.hand_terms(hand, .05, True, CFG)
        self.assertGreater(good['hand_closure'], 0.)
        self.assertEqual(R.hand_terms(hand, 1., False, CFG)['hand_closure'], 0.)
        bad = hand.copy()
        bad[0:4] = [2.2, 0., -1., 2.2]
        self.assertGreater(sum(good.values()), sum(R.hand_terms(bad, .05, True, CFG).values()))
        with self.assertRaises(ValueError):
            R.hand_terms(np.zeros(12), .05, True, CFG)

    def test_actual_reward_uses_height_and_real_16_joint_state_in_both_stages(self):
        scope = dict(np=np, math=math, DcmmCfg=CFG, interception_target=R.interception_target,
                     position_terms=R.position_terms, hand_terms=R.hand_terms,
                     quaternion_to_rotation_matrix=lambda _: np.eye(3))
        exec(compile(ast.Module(body=[method('compute_reward')], type_ignores=[]), '<reward>', 'exec'), scope)
        obj = SimpleNamespace(xpos=np.array([0., .65, .34]))
        ee = SimpleNamespace(xpos=np.array([0., .65, .3]), cvel=np.zeros(6))
        qpos = np.zeros(44)
        data = SimpleNamespace(body=lambda name: obj if name == 'object' else ee,
                               qpos=qpos, qvel=np.zeros(42), time=1.)
        env = SimpleNamespace(object_motion='roll', task='Catching', stage='tracking',
                              Dcmm=SimpleNamespace(data=data, model=SimpleNamespace(
                                  geom_size=np.array([[.04, 0., 0.]]),
                                  opt=SimpleNamespace(gravity=np.array([0., 0., -9.81])))),
                              object_name='object', object_id=0, _prev_roll_d=None,
                              terminated_reason=None, step_touch=True, reward_touch=0.,
                              start_time=0., info=dict(base_distance=0., ee_distance=.04),
                              contacts=dict(base_contacts=np.array([])), arm_limit=True,
                              norm_ctrl=lambda *_: 0., print_reward=False)
        # Deliberately use the real policy format (12-vector, not a joints dictionary).
        obs = dict(hand=np.zeros(12), arm=dict(ee_quat=np.array([1., 0., 0., 0.])),
                   object=dict(v_lin_3d=np.zeros(3)))
        for stage in ('tracking', 'grasping'):
            env.stage = stage
            info = dict(base_distance=0., ee_distance=.04, env_time=1.)
            qpos[21:37] = 0.
            env._prev_roll_d = None
            open_reward = scope['compute_reward'](env, obs, info, {})
            qpos[21 + np.array([0, 2, 3, 4, 6, 7, 8, 10, 11])] = np.tile(CFG.roll_hand_close_target, 3)
            qpos[34:37] = CFG.roll_hand_thumb_target
            env._prev_roll_d = None
            closed_reward = scope['compute_reward'](env, obs, info, {})
            self.assertGreater(closed_reward, open_reward)
            self.assertGreater(info['roll_reward_terms']['height'], 0.)
            self.assertGreater(info['roll_reward_terms']['hand_closure'], 0.)

    def test_hand_contact_enters_grasping_but_collision_has_priority(self):
        node = next(n for n in method('step').body if isinstance(n, ast.If)
                    and 'self.stage' in ast.unparse(n) and "self.task == 'Catching'" in ast.unparse(n.test))
        code = compile(ast.Module(body=[node], type_ignores=[]), '<transition>', 'exec')
        obs = dict(arm=dict(ee_pos3d=np.zeros(3), ee_quat=np.array([1., 0., 0., 0.])),
                   object=dict(pos3d=np.array([.1, .1, 0.])))
        for collision in (False, True):
            env = SimpleNamespace(task='Catching', object_motion='roll', stage='tracking',
                                  terminated=collision, contacts={'object_contacts': np.array([11])},
                                  floor_id=0, table_geom_id=1, hand_start_id=10, object_id=30,
                                  roll_palm_face_cos=.94, roll_tracking_xy_thresh=.03,
                                  roll_tracking_z_thresh=.03)
            exec(code, dict(self=env, obs=obs, np=np, quaternion_to_rotation_matrix=lambda _: np.eye(3)))
            self.assertEqual(env.stage, 'tracking' if collision else 'grasping')


if __name__ == '__main__':
    unittest.main()
