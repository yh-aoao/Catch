import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('evaluation', ROOT/'gym_dcmm/utils/roll_evaluation.py')
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)


class RollEvaluationTests(unittest.TestCase):
    def test_hold_requires_duration_and_tracks_later_drop(self):
        state = {}
        for _ in range(149): E.advance(state, True, True, True, .1, .002)
        self.assertFalse(state['caught_once'])
        E.advance(state, True, True, True, .1, .002)
        self.assertTrue(state['held_now'])
        E.advance(state, False, True, False, 1., .002)
        self.assertTrue(state['caught_once'])
        self.assertFalse(state['held_now'])

    def test_gap_does_not_earn_hold_time_and_large_gap_resets(self):
        state = {}
        E.advance(state, True, True, True, .1, .2)
        E.advance(state, False, True, True, .1, .04)
        self.assertAlmostEqual(state['duration'], .2)
        self.assertFalse(state['held_now'])
        E.advance(state, False, True, True, .1, .002)
        self.assertEqual(state['duration'], 0.)

    def test_table_and_high_speed_are_not_stable_catches(self):
        for clear, speed in [(False, .01), (True, .3)]:
            state = {}
            for _ in range(200): E.advance(state, True, clear, True, speed, .002)
            self.assertFalse(state['caught_once'])
        self.assertFalse(E.advance({}, True, False, True, .01, .4)['intercepted'])

    def test_publish_changes_no_training_state(self):
        for task in ('Tracking', 'Catching'):
            env = SimpleNamespace(object_motion='roll', task=task, step_touch=True,
                terminated_reason=None, roll_log=False,
                _roll_eval=dict(intercepted=True, caught_once=True, held_now=False))
            before = dict(env.__dict__)
            info = dict(success=False)
            E.publish(env, info, False, True)
            self.assertEqual(env.__dict__, before)
            self.assertTrue(info['roll_eval_success'])
            self.assertFalse(info['roll_final_hold'])
            self.assertEqual(info['success'], task == 'Tracking')

    def test_vector_autoreset_uses_terminal_evaluation(self):
        tree = ast.parse((ROOT/'gym_dcmm/algs/ppo_dcmm_645edc4/ppo_dcmm_track.py').read_text(encoding='utf-8'))
        method = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'terminal_metrics')
        scope = dict(np=np)
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<metrics>', 'exec'), scope)
        infos = dict(success=np.array([False, False]), _final_info=np.array([True, False]),
                     final_info=np.array([dict(success=False, roll_eval_success=True), None], dtype=object))
        success, _ = scope['terminal_metrics'](infos, np.array([True, False]), np.array([True, False]))
        np.testing.assert_array_equal(success, [True, False])


if __name__ == '__main__': unittest.main()
