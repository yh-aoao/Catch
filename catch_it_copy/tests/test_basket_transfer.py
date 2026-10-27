import copy
import unittest

import numpy as np
import torch

from gym_dcmm.algs.ppo_dcmm.models_track import ActorCritic
from gym_dcmm.algs.ppo_dcmm.utils import RunningMeanStd
from gym_dcmm.utils.basket_transfer import migrate_actor
from gym_dcmm.utils.basket_mobile import make_config, new_state, base_command
from tests.test_basket_mobile import config, parking


def model(width=35, actions=18, units=None):
    return ActorCritic(dict(actor_units=units or [16, 8], actions_num=actions,
                            input_shape=(width,), separate_value_mlp=True))


class TransferTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.source, self.target = model(), model(51, 20)
        self.source_norm = RunningMeanStd((35,)).eval()
        self.target_norm = RunningMeanStd((51,)).eval()
        self.source_norm.running_mean.copy_(torch.linspace(-1., 1., 35))
        self.source_norm.running_var.copy_(torch.linspace(.3, 2., 35))
        self.checkpoint = dict(model=copy.deepcopy(self.source.state_dict()),
            running_mean_std=copy.deepcopy(self.source_norm.state_dict()),
            action_layout='arm6_hand12', basket_arm_control='joint_velocity')

    def test_exact_actor_mean_preserved_with_shifted_parking_reference(self):
        raw = torch.randn(8, 35)
        mobile = torch.cat([raw.clone(), torch.randn(8, 16)], dim=-1)
        mobile[:, 34] -= .2
        reference = self.source.act_inference(dict(obs=self.source_norm(raw)))
        migrate_actor(self.target, self.target_norm, self.checkpoint)
        actual = self.target.act_inference(dict(obs=self.target_norm(mobile)))
        torch.testing.assert_close(actual[:, 2:], reference, atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(actual[:, :2], torch.zeros(8, 2))
        self.assertFalse(self.target_norm.training)

    def test_critic_fresh_new_inputs_zero_and_old_sigma_correctly_shifted(self):
        critic = {k: v.clone() for k, v in self.target.state_dict().items()
                  if k.startswith(('value.', 'value_mlp.'))}
        metadata = migrate_actor(self.target, self.target_norm, self.checkpoint)
        for key, value in critic.items():
            torch.testing.assert_close(self.target.state_dict()[key], value)
        self.assertEqual(torch.count_nonzero(self.target.actor_mlp.mlp[0].weight[:, 35:]), 0)
        torch.testing.assert_close(self.target.sigma[2:], self.source.sigma)
        torch.testing.assert_close(self.target.sigma[:2], torch.full((2,), -.7))
        self.assertTrue(metadata['freeze_input_normalization'])
        self.assertTrue(all(p.requires_grad for p in self.target.actor_mlp.parameters()))

    def test_invalid_source_rejected_before_model_mutation(self):
        initial = copy.deepcopy(self.target.state_dict())
        for key, value in [('action_layout', 'base2_arm6_hand12'),
                           ('basket_arm_control', 'cartesian'), ('running_mean_std', None)]:
            bad = dict(self.checkpoint, **{key: value})
            with self.assertRaises(ValueError):
                migrate_actor(self.target, self.target_norm, bad)
        for key, value in initial.items():
            torch.testing.assert_close(self.target.state_dict()[key], value)
        with self.assertRaises(ValueError):
            migrate_actor(model(51, 20, [32, 8]), self.target_norm, self.checkpoint)

    def test_braking_envelope_and_zero_command_after_unlock(self):
        cfg = config(approach_braking=True, brake_after_unlock=True)
        state = new_state(.5)
        state['physics_time'] = 1.
        near = base_command(state, [0., .5], .04, cfg, parking(.05))
        np.testing.assert_array_equal(near, [0., 0.])
        state['previous_command'] = np.array([0., .03])
        state['unlocked'] = True
        stopped = base_command(state, [0., .5], .04, cfg, parking(.08))
        np.testing.assert_array_equal(stopped, [0., 0.])
        with self.assertRaises(ValueError):
            base_command(state, [0., .5], .04, cfg)

    def test_original_mobile_mode_remains_policy_driven_after_unlock(self):
        cfg = config()
        state = new_state(.5)
        state.update(physics_time=1., unlocked=True)
        self.assertGreater(np.linalg.norm(base_command(state, [0., .5], .04, cfg)), 0.)


if __name__ == '__main__':
    unittest.main()
