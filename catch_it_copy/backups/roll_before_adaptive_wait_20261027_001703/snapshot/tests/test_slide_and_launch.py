import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET
import numpy as np
from motion_slide.geometry import SlideConfig, frame, release_pose, intercept_target

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('launch_test', ROOT/'gym_dcmm/utils/basket_launch.py')
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


class Tests(unittest.TestCase):
    def test_old_basket_weights_cannot_silently_use_new_action_semantics(self):
        tree = ast.parse((ROOT/'gym_dcmm/algs/ppo_dcmm/ppo_dcmm_catch_one_stage.py').read_text(encoding='utf-8'))
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name=='restore_test')
        checkpoint = {'model': {}}
        scope = dict(torch=SimpleNamespace(load=lambda *args, **kwargs: checkpoint))
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<restore>', 'exec'), scope)
        loaded = []
        env = SimpleNamespace(device='cpu', fixed_basket=True, basket_arm_control='joint_velocity',
                              normalize_input=False, _load_compatible_model_state=loaded.append)
        with self.assertRaises(ValueError):
            scope['restore_test'](env, 'old.pth')
        self.assertFalse(loaded)
        checkpoint['basket_arm_control'] = 'joint_velocity'
        scope['restore_test'](env, 'new.pth')
        self.assertEqual(len(loaded), 1)

    def test_slide_success_is_after_edge_and_failure_overrides_touch(self):
        tree = ast.parse((ROOT/'motion_slide/environment.py').read_text(encoding='utf-8'))
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name=='step')
        sim = SimpleNamespace(mj_step=lambda model, data: setattr(data, 'time', data.time+.01),
                              mj_rnePostConstraint=lambda *args: None, mj_forward=lambda *args: None)
        scope = dict(np=np, mujoco=sim, hand_collision_ids=lambda *args: np.array([1]), intercept_target=intercept_target)
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<slide-step>', 'exec'), scope)
        def make(task):
            data = SimpleNamespace(qpos=np.zeros(44), qvel=np.zeros(42), ctrl=np.zeros(31),
                                   geom_xpos=np.zeros((12, 3)), time=0.)
            data.qpos[37:40] = [0., .5, .53]
            data.geom_xpos[1] = [0., .5, .5]
            model = SimpleNamespace(opt=SimpleNamespace(timestep=.01), jnt_qposadr=np.arange(44), jnt_limited=np.zeros(44, bool))
            robot = SimpleNamespace(model=model, data=data, target_base_vel=np.zeros(3), target_arm_qpos=np.zeros(6),
                                    target_hand_qpos=np.zeros(16), move_ee_pose=lambda *a, **kw: ((np.zeros(6), True), None),
                                    action_hand2qpos=lambda *a: None)
            contacts = dict(object_contacts=np.array([1]), base_contacts=np.array([]), hand_contacts=np.array([]))
            return SimpleNamespace(Dcmm=robot, task=task, steps=0, steps_per_policy=20, start_time=0., env_time=5.,
                                   slide=SlideConfig(), slide_log=False, departed=False, caught=False, hold=0., gap=0.,
                                   previous_action=None, hand_start_id=1, table_geom_id=2, floor_id=9,
                                   update_target_ctrl=lambda: None, _get_ctrl=lambda: np.zeros(30),
                                   _get_contacts=lambda: contacts, _get_info=lambda: {}, _get_obs=lambda: {}, render=lambda: None)
        action = dict(base=np.zeros(2), arm=np.zeros(6), hand=np.zeros(12))
        env = make('Tracking')
        result = scope['step'](env, action)
        self.assertTrue(result[4]['success'])
        env = make('Tracking')
        env.Dcmm.data.qpos[38] = 1.1  # Touching while still above ramp does not count.
        self.assertFalse(scope['step'](env, action)[4]['success'])
        env = make('Tracking')
        env._get_contacts()['object_contacts'] = np.array([1, 9])
        result = scope['step'](env, action)
        self.assertFalse(result[4]['success'])
        self.assertEqual(result[4]['terminated_reason'], 'object_on_floor')
        env = make('Catching')
        self.assertFalse(scope['step'](env, action)[4]['success'])
        self.assertTrue(scope['step'](env, action)[4]['success'])

    def test_ramp_edge_release_and_gravity(self):
        cfg = SlideConfig().validate()
        edge, center, rotation, quat = frame(cfg)
        np.testing.assert_allclose(center+rotation@np.array([0., -cfg.length/2, cfg.thickness/2]), edge)
        position, _ = release_pose(cfg)
        self.assertGreater(position[1], edge[1])
        self.assertGreater(position[2], edge[2])
        self.assertGreater(np.tan(np.radians(cfg.angle_deg)), cfg.friction)
        target = intercept_target(position, np.zeros(3), cfg, True)
        self.assertLess(target[1], edge[1]-.1)
        self.assertEqual(target[2], cfg.wait_z)
        np.testing.assert_allclose(rotation.T@rotation, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(quat), 1.)
        with self.assertRaises(ValueError):
            SlideConfig(friction=1.).validate()

    def test_slide_primitive_inertia_and_no_mesh(self):
        tree = ast.parse((ROOT/'motion_slide/environment.py').read_text(encoding='utf-8'))
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name=='_reset_object_roll')
        scope = dict(ET=ET)
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<slide-xml>', 'exec'), scope)
        env = SimpleNamespace(slide=SlideConfig(), Dcmm=SimpleNamespace(model_xml_string='''<mujoco><worldbody><body name="object"><inertial mass="1" pos="0 0 0" diaginertia="1 1 1"/><joint name="object" type="free"/><geom name="object" type="mesh" mesh="old"/></body></worldbody></mujoco>'''))
        result = ET.fromstring(scope['_reset_object_roll'](env))
        self.assertIsNone(result.find('.//inertial'))
        geom = result.find('.//geom')
        self.assertEqual(geom.get('type'), 'box')
        self.assertIsNone(geom.get('mesh'))
        self.assertEqual(float(geom.get('mass')), .05)

    def test_joint_action_bounds_and_physical_time(self):
        limits = np.tile([-1., 1.], (6, 1))
        result = launch.joint_velocity_target(np.zeros(6), np.ones(6)*.025, .025, np.ones(6)*3, .04, limits)
        np.testing.assert_allclose(result, .12)
        result = launch.joint_velocity_target(np.ones(6)*.99, np.ones(6), .025, np.ones(6)*3, .04, limits)
        np.testing.assert_allclose(result, 1.)
        result = launch.joint_velocity_target(np.zeros(6), np.zeros(6), .025, np.ones(6)*3, .04, limits)
        np.testing.assert_allclose(result, 0.)

    def test_acceleration_credit_cannot_be_farmed(self):
        p, center, reference = np.array([0., 0., .6]), np.array([0., 2., .9]), np.array([0., 3., 3.])
        a = launch.acceleration_readiness([0., .5, .5], reference, p, center)
        b = launch.acceleration_readiness(reference, reference, p, center)
        self.assertGreater(a, 0.)
        self.assertGreater(b, a)
        self.assertEqual(launch.acceleration_readiness([0., -3., -3.], reference, p, center), 0.)
        state = {}
        total = sum(launch.best_progress(state, 'speed', v) for v in [a, 0., a, b, 0., b])
        self.assertAlmostEqual(total, 1.)


if __name__ == '__main__':
    unittest.main()
