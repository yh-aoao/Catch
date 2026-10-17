import types
import unittest
import numpy as np
import torch
from distillation_event.events import ContactEvents, sample_positions


class Tests(unittest.TestCase):
    def test_substep_contact_survives_policy_step_and_reset(self):
        env = types.SimpleNamespace(table_geom_id=2, floor_id=0, hand_start_id=5, object_id=10,
                                    Dcmm=types.SimpleNamespace(data=types.SimpleNamespace(time=0.)))
        env.ids = [2]
        env._get_contacts = lambda: {'object_contacts': np.array(env.ids)}
        observer = ContactEvents(env)
        env._get_contacts()
        observer.begin_step()
        env.ids = []
        env._get_contacts()
        env.ids = [0]
        env._get_contacts()
        env.ids = [7]
        env._get_contacts()
        env.Dcmm.data.time = .13
        env._get_contacts()
        env.ids = []
        env._get_contacts()
        np.testing.assert_array_equal(observer.flags, [1, 1, 1, 1])
        observer.reset()
        np.testing.assert_array_equal(observer.flags, [0, 0, 0, 0])
        env.ids = [11]  # A later scene geom is not a finger.
        env._get_contacts()
        self.assertFalse(observer.hand_seen)

    def test_stratification_budget_padding_and_no_event_fallback(self):
        torch.manual_seed(3)
        flags = torch.zeros(2, 100, 4)
        flags[0, 50, 2] = 1
        valid = torch.ones(2, 100)
        valid[1, 60:] = 0
        cfg = dict(enabled=True, pre_steps=2, post_steps=2, samples_per_episode=1000, event_fraction=.5)
        counts = sample_positions(flags, valid, cfg)
        torch.testing.assert_close(counts.sum(1), torch.tensor([1000., 1000.]))
        self.assertGreaterEqual(counts[0, 48:53].sum(), 500.)
        self.assertEqual(counts[1, 60:].sum(), 0.)
        cfg['enabled'] = False
        torch.testing.assert_close(sample_positions(flags, valid, cfg), valid)


if __name__ == '__main__':
    unittest.main()
