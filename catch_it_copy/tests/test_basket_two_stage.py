import copy
import unittest

import numpy as np
import torch

from gym_dcmm.algs.ppo_dcmm.models_track import ActorCritic
from gym_dcmm.algs.ppo_dcmm.utils import RunningMeanStd
from gym_dcmm.utils.basket_parking_policy import FrozenParkingPolicy
from gym_dcmm.utils.basket_transfer import migrate_actor
from gym_dcmm.utils import basket_mobile as mobile
from tests.test_basket_mobile import config, parking
from tests.test_basket_fixed_policy import SCOPE, ArrayTensor
from types import SimpleNamespace as N


class TwoStageTests(unittest.TestCase):
    def setUp(self):
        self.nav = ActorCritic(dict(actor_units=[16, 8], actions_num=2,
                                   input_shape=(51,), separate_value_mlp=True))
        self.norm = RunningMeanStd((51,)).eval()
        self.checkpoint = dict(model=self.nav.state_dict(), running_mean_std=self.norm.state_dict(),
            basket_training_stage='parking', action_layout='base2_parking',
            basket_observation_version='mobile51_v1', basket_mobile_options=dict(parking_only=True))

    def test_frozen_parking_keeps_own_statistics_and_stops_after_unlock(self):
        frozen = FrozenParkingPolicy(self.checkpoint, dict(parking_only=False), 'cpu')
        self.assertTrue(all(not p.requires_grad for p in frozen.model.parameters()))
        self.assertFalse(frozen.model.training)
        self.assertFalse(frozen.normalizer.training)
        stats = copy.deepcopy(frozen.normalizer.state_dict())
        raw = torch.randn(4, 51);raw[:, 47] = 0.
        expected = self.nav.act_inference(dict(obs=self.norm(raw))).clamp(-1.,1.)
        torch.testing.assert_close(frozen.actions(raw), expected)
        raw[:, 47] = 1.
        torch.testing.assert_close(frozen.actions(raw), torch.zeros(4, 2))
        for key, value in stats.items():
            torch.testing.assert_close(frozen.normalizer.state_dict()[key], value)
        loaded = FrozenParkingPolicy(frozen.export(), dict(parking_only=False), 'cpu')
        torch.testing.assert_close(loaded.actions(raw), frozen.actions(raw))

    def test_parking_rejects_joint_weights_and_geometry_mismatch(self):
        with self.assertRaises(ValueError):
            FrozenParkingPolicy(dict(self.checkpoint, action_layout='base2_arm6_hand12'), {}, 'cpu')
        with self.assertRaises(ValueError):
            FrozenParkingPolicy(self.checkpoint, dict(max_speed=3.), 'cpu')

    def test_stage_action_mapping_has_no_shift_between_base_and_arm(self):
        source = np.arange(40, dtype=float).reshape(2,20)/40.
        base_policy = N(mobile_basket=True, basket_training_stage='parking', mobile_base_scale=.5)
        result = SCOPE['action2dict'](base_policy, ArrayTensor(source))
        np.testing.assert_allclose(result['base'], source[:, :2]*.5)
        np.testing.assert_array_equal(result['arm'], np.zeros((2,6)))
        throw_policy = N(mobile_basket=True, basket_training_stage='throw', mobile_base_scale=.5,
            obs=dict(obs=torch.zeros(2,51)), action_catch_denorm=[1.5,.025,.15],
            parking_policy=N(actions=lambda raw: torch.full((len(raw),2), .2)))
        result = SCOPE['action2dict'](throw_policy, ArrayTensor(source))
        np.testing.assert_allclose(result['base'], .1)
        np.testing.assert_allclose(result['arm'], source[:, :6]*.025)
        np.testing.assert_allclose(result['hand'], source[:, 6:18]*.15)

    def test_parking_success_does_not_pay_unreleased_or_task_failure(self):
        state = mobile.new_state(.5)
        state.update(unlocked=True, ready_pending=3.)
        value, terms = mobile.reward(state, {}, parking(), True, True, 1., config(parking_only=True))
        self.assertGreater(value, 0.)
        self.assertEqual(terms['failure'], 0.)
        self.assertEqual(terms['task_failure'], 0.)
        self.assertEqual(terms['basket_hit'], 0.)

    def test_arm18_initialization_preserves_source_output_in_stage_two(self):
        source = ActorCritic(dict(actor_units=[16,8], actions_num=18,
            input_shape=(35,), separate_value_mlp=True))
        target = ActorCritic(dict(actor_units=[16,8], actions_num=18,
            input_shape=(51,), separate_value_mlp=True))
        src_norm, dst_norm = RunningMeanStd((35,)).eval(), RunningMeanStd((51,)).eval()
        raw = torch.randn(4,35);extended = torch.cat((raw, torch.randn(4,16)),dim=-1)
        extended[:,34] -= .2
        migrate_actor(target,dst_norm,dict(model=source.state_dict(),
            running_mean_std=src_norm.state_dict(),action_layout='arm6_hand12',basket_arm_control='joint_velocity'))
        torch.testing.assert_close(source.act_inference(dict(obs=src_norm(raw))),
                                   target.act_inference(dict(obs=dst_norm(extended))))


if __name__ == '__main__':
    unittest.main()
