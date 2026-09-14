"""CPU regression tests for basket events, real phase code and frozen PPO actions."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace, MethodType
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


C = load('basket_cfg_test', 'configs/env/DcmmCfg.py')
B = load('basket_catching_test', 'gym_dcmm/utils/basket_catching.py')
P = load('basket_parking_test', 'gym_dcmm/utils/basket_tracking.py')
M = load('basket_model_test', 'gym_dcmm/algs/ppo_dcmm/models_catch.py')
tree = ast.parse((ROOT / 'gym_dcmm/envs/DcmmVecEnv.py').read_text(encoding='utf-8'))
scope = dict(np=np, DcmmCfg=C, limit_speed=P.limit_speed,
             **{n: getattr(B, n) for n in ('hoop_crossing', 'flight_failure', 'predicted_miss', 'catching_reward')})
for name in ('_basket_hold_and_release', '_basket_check_flight', 'compute_reward'):
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<actual-basket>', 'exec'), scope)


def fixture():
    ee = SimpleNamespace(xpos=np.array([0., .4, .45]), xmat=np.eye(3))
    obj = SimpleNamespace(xpos=np.array([0., .4, .55]))
    data = SimpleNamespace(time=0., ctrl=np.zeros(1), body=lambda n: obj if n == 'object' else ee)
    parking = dict(settled=False, in_position=False, distance=.2,
                   velocity=np.zeros(2), desired_velocity=np.zeros(2))
    calls = []
    env = SimpleNamespace(task='Catching', object_motion='throw_basket', object_name='object',
        Dcmm=SimpleNamespace(data=data, model=SimpleNamespace(opt=SimpleNamespace(timestep=.01, gravity=np.array([0., 0., -9.81]))),
                             set_throw_pos_vel=lambda **kw: calls.append(kw)),
        object_throw=False, basket_phase='parking', basket_settled_time=0., basket_prepare_start=None,
        basket_hold_previous=None, basket_hold_velocity=np.zeros(3), object_q=np.array([1., 0., 0., 0.]),
        random_mass=.05, steps_per_policy=4, basket_center=np.array([0., 2.2, .9]),
        basket_previous_distance=.2, basket_previous_flight_distance=None,
        terminated=False, terminated_reason=None, floor_id=9, contacts={'object_contacts': np.array([])},
        _basket_parking_state=lambda: parking)
    env.hold = MethodType(scope['_basket_hold_and_release'], env)
    return env, parking, calls


class BasketTests(unittest.TestCase):
    def test_crossing_checks_direction_intersection_radius_and_translation(self):
        center = np.array([1., 2., .9])
        for offset in (np.zeros(3), np.array([4., -3., .2])):
            c = center + offset
            self.assertEqual(B.hoop_crossing(c+[.1, 0., .3], c+[.1, 0., -.3], c, 0., .2, .04)[:2], (True, True))
            self.assertEqual(B.hoop_crossing(c+[.18, 0., .3], c+[.18, 0., -.3], c, 0., .2, .04)[:2], (True, False))
            self.assertFalse(B.hoop_crossing(c-[0., 0., .3], c+[0., 0., .3], c, 0., .2, .04)[0])
            self.assertFalse(B.hoop_crossing(c, c, c, 0., .2, .04)[0])

    def test_tilted_hoop_and_floor_failure_priority(self):
        env, _, _ = fixture()
        normal = np.array([0., -np.sin(np.radians(C.basket_tilt_deg)), np.cos(np.radians(C.basket_tilt_deg))])
        previous = env.basket_center + .2 * normal
        env.Dcmm.data.body('object').xpos[:] = env.basket_center - .2 * normal
        scope['_basket_check_flight'](env, previous)
        self.assertEqual(env.terminated_reason, 'basket_score')
        env.contacts['object_contacts'] = np.array([env.floor_id])
        scope['_basket_check_flight'](env, previous)
        self.assertEqual(env.terminated_reason, 'ball_on_floor')

    def test_release_waits_for_parking_and_keeps_attachment_point(self):
        env, parking, calls = fixture()
        for i in range(100):
            env.Dcmm.data.time = i * .01
            env.hold()
        self.assertFalse(env.object_throw)
        parking.update(settled=True, in_position=True)
        for i in range(20):
            env.Dcmm.data.time += .01
            env.hold()
        self.assertEqual(env.basket_phase, 'preparing')
        before = calls[-1]['pose'].copy()
        env.Dcmm.data.time = env.basket_prepare_start + C.basket_hold_duration + .01
        env.hold()
        self.assertTrue(env.object_throw)
        self.assertEqual(env.basket_phase, 'flight')
        np.testing.assert_allclose(calls[-1]['pose'], before)
        np.testing.assert_allclose(calls[-1]['velocity'][:3], C.basket_release_boost)
        self.assertEqual(env.Dcmm.data.ctrl[-1], 0.)

    def test_leaving_parking_resets_preparation(self):
        env, parking, _ = fixture()
        env.basket_phase = 'preparing'
        env.basket_prepare_start = 0.
        env.Dcmm.data.time = 1.
        env.hold()
        self.assertEqual(env.basket_phase, 'parking')
        self.assertIsNone(env.basket_prepare_start)
        self.assertFalse(env.object_throw)

    def test_actual_reward_uses_world_position_and_cannot_farm_idle_flight(self):
        env, _, _ = fixture()
        env.basket_phase = 'flight'
        env.object_throw = True
        d = np.linalg.norm(env.Dcmm.data.body('object').xpos - env.basket_center)
        env.basket_previous_flight_distance = d
        info = dict(success=False, env_time=1.)
        result = scope['compute_reward'](env, {'object': {'pos3d': np.array([99., 99., 99.])}}, info, {})
        self.assertAlmostEqual(info['basket_distance_world'], d)
        self.assertAlmostEqual(result, -C.basket_catch_time_cost)
        info.update(success=True)
        env.terminated = True
        result = scope['compute_reward'](env, {}, info, {})
        self.assertAlmostEqual(result, C.basket_w_score - C.basket_catch_time_cost)
        self.assertEqual(info['basket_reward_terms']['failure'], 0.)

    def test_preparation_rewards_better_trajectory_not_hand_pose(self):
        env, parking, _ = fixture()
        controls = dict(arm=np.zeros(6), hand=np.zeros(12))
        def value(error):
            return B.catching_reward('preparing', parking, .2, 1., None, error, controls, False, False, C)[0]
        self.assertGreater(value(.1), value(1.))
        self.assertLess(value(.1), 0.)

    def test_frozen_base_actions_and_likelihood_match_training(self):
        model = M.ActorCritic(dict(separate_value_mlp=True, actions_num=20,
            tracking_actions_num=2, input_shape=(35,), actor_units=[16], freeze_tracking=True))
        obs = dict(obs=torch.randn(4, 35), obs_t=torch.randn(4, 23), obs_c=torch.randn(4, 33))
        sample = model.act(obs)
        torch.testing.assert_close(sample['actions'][:, :2], sample['mus'][:, :2])
        before = model.act_inference(obs)[:, :2].clone()
        result = model(dict(obs, prev_actions=sample['actions']))
        torch.testing.assert_close(result['prev_neglogp'], sample['neglogpacs'])
        expected_entropy = 18 * .5 * (1. + np.log(2. * np.pi))
        torch.testing.assert_close(result['entropy'], torch.full((4,), expected_entropy))
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        (result['prev_neglogp'].mean() + result['values'].square().mean()).backward()
        self.assertIsNone(model.mu_t.weight.grad)
        self.assertIsNotNone(model.mu_c.weight.grad)
        optimizer.step()
        torch.testing.assert_close(model.act_inference(obs)[:, :2], before)

    def test_frozen_tracking_normalizer_stays_in_eval_mode(self):
        source = ast.parse((ROOT / 'gym_dcmm/algs/ppo_dcmm/ppo_dcmm_catch_two_stage.py').read_text(encoding='utf-8'))
        method = next(n for n in ast.walk(source) if isinstance(n, ast.FunctionDef) and n.name == 'set_train')
        local = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<actual-normalization>', 'exec'), local)
        agent = SimpleNamespace(model=torch.nn.Identity(), normalize_input=True, normalize_value=True,
            freeze_tracking=True, running_mean_std_track=torch.nn.Identity(),
            running_mean_std_hand=torch.nn.Identity(), value_mean_std=torch.nn.Identity())
        local['set_train'](agent)
        self.assertFalse(agent.running_mean_std_track.training)
        self.assertTrue(agent.running_mean_std_hand.training)
        agent.freeze_tracking = False
        local['set_train'](agent)
        self.assertTrue(agent.running_mean_std_track.training)

    def test_basket_holding_does_not_use_incoming_ball_y_bound(self):
        simulation = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_step_mujoco_simulation')
        bounds = next(n for n in ast.walk(simulation) if isinstance(n, ast.If)
            and ast.unparse(n.test) == "self.object_motion == 'roll'"
            and any(isinstance(child, ast.Name) and child.id == 'off_table_back' for child in ast.walk(n)))
        local = dict(self=SimpleNamespace(object_motion='throw_basket'), obj_y=.4, obj_x=0., obj_z=.5, obj_radius=.04)
        exec(compile(ast.Module(body=[bounds], type_ignores=[]), '<actual-bounds>', 'exec'), local)
        self.assertFalse(local['out_of_bounds'])


if __name__ == '__main__':
    unittest.main()
