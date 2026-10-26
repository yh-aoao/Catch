"""CPU-only checks: presets plus the actual environment launch/physics code.

Run: python -m unittest discover -s tests -p test_bounce_presets.py -v
These checks do not validate MuJoCo trajectories or catching feasibility.
"""
import ast
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import configs.env.DcmmCfg as cfg
from configs.env.bounce_presets import (
    PHYSICS_BASE, PHYSICS_GROUPS, LAUNCH_GROUPS, episode_config, parse_groups,
)


tree = ast.parse((ROOT / 'gym_dcmm/envs/DcmmVecEnv.py').read_text(encoding='utf-8'))
env_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DcmmVecEnv')
pose_node = next(n for n in env_class.body if isinstance(n, ast.FunctionDef)
                 and n.name == 'random_object_pose_bounce')
namespace = dict(np=np, math=math)
exec(compile(ast.Module(body=[pose_node], type_ignores=[]), '<actual pose method>', 'exec'), namespace)
pose = namespace['random_object_pose_bounce']
reset_node = next(n for n in env_class.body if isinstance(n, ast.FunctionDef)
                  and n.name == '_reset_simulation')
physics_node = next(n for n in reset_node.body if isinstance(n, ast.If)
                    and any(isinstance(a, ast.Assign) and any(
                        isinstance(t, ast.Attribute) and t.attr == 'bounce_restitution'
                        for t in a.targets) for a in n.body))
physics_code = compile(ast.Module(body=[physics_node], type_ignores=[]),
                       '<actual physics reset block>', 'exec')


class PresetTests(unittest.TestCase):
    def make_env(self, physics, launch):
        params, launch_values, p, l = episode_config(cfg, [physics], [launch], np.random)
        return SimpleNamespace(
            bounce_cfg=params, bounce_launch_values=launch_values,
            bounce_physics_group=p, bounce_launch_group=l,
            # Deliberately stale radius: launch must use this episode's preset.
            Dcmm=SimpleNamespace(model=SimpleNamespace(geom_size=np.array([[0.1, 0, 0]]))),
            object_id=0, print_info=False,
        )

    def test_all_fixed_groups_and_actual_launch(self):
        for p in PHYSICS_GROUPS:
            for l in ['L0', 'L1', 'L2', 'L3', 'L4']:
                env = self.make_env(p, l)
                pose(env)
                expected = dict(PHYSICS_BASE, **PHYSICS_GROUPS[p])
                self.assertEqual(env.bounce_cfg.bounce_mass, [expected['mass']] * 2)
                np.testing.assert_allclose(env.object_pos3d, [0, 2.4, 0.9])
                speed = {'L1': 0.95, 'L2': 1.05}.get(l, 1.0)
                angle = math.radians({'L3': -10, 'L4': 10}.get(l, 0))
                np.testing.assert_allclose(env.object_vel6d[:3],
                                           [speed * math.sin(angle), -speed * math.cos(angle), -0.2])
                self.assertAlmostEqual(np.linalg.norm(env.object_vel6d[3:5]), speed / expected['radius'])

    def test_mixture_reproducible_and_isolated(self):
        def sample():
            rng = np.random.RandomState(123)
            return [episode_config(cfg, ['P0', 'P1', 'P8'], ['L0'], rng)[2] for _ in range(200)]
        self.assertEqual(sample(), sample())
        self.assertEqual(set(sample()), {'P0', 'P1', 'P8'})
        env = self.make_env('P1', 'gentle')
        env.bounce_cfg.bounce_hand_friction[0][0] = 9
        self.assertEqual(PHYSICS_BASE['hand_friction'][0], 2.0)
        self.assertEqual(float(cfg.bounce_radius[0]), 0.04)
        for _ in range(50):
            pose(env)
            self.assertTrue(0.95 <= np.linalg.norm(env.object_vel6d[:2]) <= 1.05)

    def test_actual_physics_reset_writes_selected_values(self):
        mock_mujoco = SimpleNamespace(
            mjtObj=SimpleNamespace(mjOBJ_BODY=1, mjOBJ_JOINT=2, mjOBJ_GEOM=3),
            mj_name2id=lambda model, kind, name: 2 if kind == 3 else 0,
        )
        for p in list(PHYSICS_GROUPS) + ['legacy']:
            env = self.make_env(p, 'L0')
            env.object_motion = 'bounce'
            env.floor_id = 1
            env.print_bounce_info = False
            model = env.Dcmm.model
            model.geom_size = np.zeros((3, 3))
            model.geom_solref = np.zeros((3, 2))
            model.geom_solimp = np.zeros((3, 5))
            model.geom_friction = np.zeros((3, 3))
            model.body_mass = np.zeros(1)
            model.body_inertia = np.zeros((1, 3))
            model.jnt_dofadr = np.array([0])
            model.dof_damping = np.zeros(6)
            exec(physics_code, dict(self=env, np=np, mujoco=mock_mujoco))
            expected = dict(PHYSICS_BASE, **PHYSICS_GROUPS.get(p, {}))
            self.assertEqual(model.geom_size[0, 0], expected['radius'])
            self.assertEqual(model.body_mass[0], expected['mass'])
            np.testing.assert_allclose(model.body_inertia[0],
                                       7.33516e-05 if p == 'legacy' else
                                       0.4 * expected['mass'] * expected['radius'] ** 2)
            np.testing.assert_allclose(model.dof_damping,
                                       0.00015 if p == 'legacy' else expected['joint_damping'])
            np.testing.assert_allclose(model.geom_friction[0], expected['friction'])
            np.testing.assert_allclose(model.geom_friction[2], expected['hand_friction'])
            np.testing.assert_allclose(model.geom_solref[0],
                                       [0.008, (1 - expected['restitution']) * 0.5])
            np.testing.assert_allclose(model.geom_solref[1], model.geom_solref[0])

    def test_legacy_rng_and_ranges(self):
        np.random.seed(91)
        before = np.random.get_state()
        params, launch, _, _ = episode_config(cfg, ['legacy'], ['legacy'], np.random)
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2], after[2])
        np.testing.assert_array_equal(params.bounce_mass, cfg.bounce_mass)
        self.assertEqual(launch['angle_deg'], [-25, 25])
        self.assertEqual(launch['spin_factor'], [0.5, 1.5])
        self.assertEqual(launch['y'], [2.1, 2.7])

    def test_invalid_selection(self):
        for value in ['', 'P9', 'P0,', 'P0,P0', ['P0']]:
            with self.assertRaises(ValueError):
                parse_groups(value, PHYSICS_GROUPS)
        self.assertEqual(parse_groups('P0, P2', PHYSICS_GROUPS), ['P0', 'P2'])


if __name__ == '__main__':
    unittest.main()
