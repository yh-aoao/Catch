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

    def test_open_before_arrival_even_if_finger_touched(self):
        self.assertFalse(self.reward(.8, [.2, .05, 0], True)[0])
        self.assertGreater(self.reward(.15, [0, .3, 0])[1], self.reward(.8, [0, .3, 0])[1])

    def test_close_inside_palm_and_avoid_overflexion(self):
        self.assertTrue(self.reward(.8, [0, .05, 0])[0])
        self.assertGreater(self.reward(.8, [0, .05, 0])[1], self.reward(.15, [0, .05, 0])[1])
        self.assertGreater(self.reward(.8, [0, .05, 0])[1], self.reward(2., [0, .05, 0])[1])


if __name__ == '__main__':
    unittest.main()
