import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('grasp', root / 'gym_dcmm/utils/roll_grasp.py')
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
cfg = SimpleNamespace(roll_grasp_entry_radius=.09, roll_grasp_entry_depth=.12,
    roll_grasp_closed_target=.8, roll_grasp_open_target=.15,
    roll_grasp_posture_weight=2., roll_grasp_chain_weight=.5)


class GraspTests(unittest.TestCase):
    def reward(self, angle, point, contact=False):
        ready, terms = g.grasp_terms(np.full(16, angle), point, contact, cfg)
        return ready, sum(terms.values())

    def test_shaking_cost_and_boundary_continuity(self):
        c = SimpleNamespace(roll_hand_speed_weight=.4, roll_hand_accel_weight=.2)
        still = g.hand_stability_terms(np.zeros(16), np.zeros(16), .04, True, c)
        shake = g.hand_stability_terms(np.ones(16)*3, -np.ones(16)*3, .04, True, c)
        self.assertLess(sum(shake.values()), sum(still.values()))
        near = self.reward(.5, [.09-1e-6, .05, 0])[1]
        far = self.reward(.5, [.09+1e-6, .05, 0])[1]
        self.assertLess(abs(near-far), .001)

    def test_target_filter_is_bounded_and_faster_before_capture(self):
        c = SimpleNamespace(roll_target_capture_rate=3., roll_target_hold_rate=1.,
            roll_target_capture_alpha=.8, roll_target_hold_alpha=.3,
            roll_target_motion_weight=.03, roll_target_change_weight=.15)
        fast, _ = g.smooth_target_delta(np.ones(12), None, .04, False, c)
        slow, _ = g.smooth_target_delta(np.ones(12), None, .04, True, c)
        self.assertTrue(np.all(fast > slow))
        self.assertTrue(np.all(slow <= .04))
        reverse, terms = g.smooth_target_delta(-np.ones(12), slow, .04, True, c)
        self.assertTrue(np.all(np.abs(reverse) <= .04))
        self.assertLess(terms['target_change'], 0.)

    def test_wait_target_stays_outside_table_even_for_low_ball(self):
        spec = importlib.util.spec_from_file_location('waiting', root / 'gym_dcmm/utils/roll_waiting.py')
        w = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(w)
        c = SimpleNamespace(roll_table_pos=[0, 2., .4], roll_table_size=[1, 1., .1],
            roll_wait_z=.5, roll_table_height=.4, roll_wait_edge_margin=.2)
        point, waiting = w.target([0, 1.2, .45], [0, -1., 0], .04, c)
        self.assertTrue(waiting)
        self.assertLessEqual(point[1], .8)

    def test_hold_mode_survives_brief_contact_gap_but_not_large_slip(self):
        import ast
        tree = ast.parse((root / 'gym_dcmm/envs/DcmmVecEnv_roll_645edc4.py').read_text(encoding='utf-8'))
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_roll_settled_mode')
        scope = dict(DcmmCfg=SimpleNamespace(roll_hold_hysteresis_seconds=.12))
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<hold>', 'exec'), scope)
        env = SimpleNamespace(Dcmm=SimpleNamespace(data=SimpleNamespace(time=1.)))
        hold = scope['_roll_settled_mode']
        self.assertTrue(hold(env, True, True, .1))
        env.Dcmm.data.time = 1.05
        self.assertTrue(hold(env, False, True, .2))
        self.assertFalse(hold(env, True, True, .5))

    def test_open_before_arrival_even_if_finger_touched(self):
        self.assertFalse(self.reward(.8, [.2, .05, 0], True)[0])
        self.assertGreater(self.reward(.15, [0, .3, 0])[1], self.reward(.8, [0, .3, 0])[1])

    def test_close_inside_palm_and_avoid_overflexion(self):
        self.assertTrue(self.reward(.8, [0, .05, 0])[0])
        self.assertGreater(self.reward(.8, [0, .05, 0])[1], self.reward(.15, [0, .05, 0])[1])
        self.assertGreater(self.reward(.8, [0, .05, 0])[1], self.reward(2., [0, .05, 0])[1])


if __name__ == '__main__':
    unittest.main()
