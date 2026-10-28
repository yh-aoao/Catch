"""Behavior tests for pretrained Track preservation and resumable Catch PPO."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import torch

from gym_dcmm.algs.ppo_dcmm_5fe75d5f.ppo_dcmm_catch_hold import (
    PPO_Catch_Hold, training_options)
from gym_dcmm.algs.ppo_dcmm_july22.models_track import ActorCritic as TrackModel
from gym_dcmm.algs.ppo_dcmm_5fe75d5f.utils import RunningMeanStd


class FakeEnvironment:
    def call(self, key):
        return [dict(act_c_dim=20, actions_low=-np.ones(20), actions_high=np.ones(20),
                     obs_c_dim=30, obs_t_dim=18, task='Catching', object_motion='bounce')[key]]

    def reset(self):
        n = 2
        obs = dict(base=dict(v_lin_2d=np.zeros((n, 2))),
                   arm=dict(ee_pos3d=np.zeros((n, 3)), ee_quat=np.zeros((n, 4)),
                            ee_v_lin_3d=np.zeros((n, 3))),
                   object=dict(pos3d=np.zeros((n, 3)), v_lin_3d=np.zeros((n, 3))),
                   hand=np.zeros((n, 12)))
        return obs, {}


class HoldPolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(37)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.track = TrackModel(dict(actor_units=[16, 8], actions_num=8,
                                    input_shape=(18,), separate_value_mlp=True))
        self.normalizer = RunningMeanStd((18,)).eval()
        self.normalizer.running_mean.copy_(torch.linspace(-1, 1, 18))
        self.normalizer.running_var.fill_(2.)
        self.normalizer.count.fill_(12345.)
        self.checkpoint = self.path/'track.pth'
        torch.save(dict(tracking_mlp=self.track.actor_mlp.state_dict(),
                        tracking_mu=self.track.mu.state_dict(),
                        tracking_sigma=self.track.sigma.detach(),
                        running_mean_std=self.normalizer.state_dict()), self.checkpoint)

    def config(self, track=True):
        train_path = Path(__file__).resolve().parents[1]/'configs/train/DcmmPPO.yaml'
        cfg = OmegaConf.create(dict(train=OmegaConf.load(train_path), rl_device='cpu',
                                    test=False, num_envs=2, seed=0,
                                    checkpoint_tracking=str(self.checkpoint) if track else '',
                                    checkpoint_catching='',
                                    bounce_grasp_training=dict(track_lr_scale=.1,
                                        freeze_track_steps=50, freeze_track_normalization=True)))
        cfg.train.network.mlp.units = [16, 8]
        cfg.train.ppo.num_actors = 2
        cfg.train.ppo.horizon_length = 4
        cfg.train.ppo.minibatch_size = 8
        cfg.train.ppo.max_agent_steps = 100
        cfg.train.ppo.save_frequency = 0
        return cfg

    def agent(self, cfg=None, suffix='agent'):
        obj = PPO_Catch_Hold(FakeEnvironment(), str(self.path/suffix), cfg or self.config())
        self.addCleanup(obj.writer.close)
        return obj

    def optimize(self, agent):
        agent.set_train()
        raw = torch.randn(8, 30)
        obs_t = agent.running_mean_std_track(raw[:, :18])
        hand = agent.running_mean_std_hand(raw[:, 18:])
        obs = torch.cat((obs_t, hand), dim=1)
        output = agent.model(dict(obs=obs, obs_t=obs_t, obs_c=obs[:, 2:],
                                  prev_actions=torch.full((8, 20), .4)))
        loss = output['prev_neglogp'].mean() + output['values'].square().mean()
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()

    def test_track_initialization_preserves_actions_and_normalization(self):
        agent = self.agent()
        agent.set_eval()
        raw = torch.randn(7, 30)
        expected = self.track.act_inference(dict(obs=self.normalizer(raw[:, :18])))
        actual = agent.model_act(dict(obs=raw), inference=True)['actions'][:, :8]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for key, val in self.normalizer.state_dict().items():
            torch.testing.assert_close(agent.running_mean_std_track.state_dict()[key], val)
        self.assertTrue(all(p.requires_grad for p in agent.model.actor_mlp_c.parameters()))
        self.assertTrue(all(not p.requires_grad for p in agent._track_parameters))

    def test_freeze_then_low_rate_update_survives_legacy_scheduler_writes(self):
        agent = self.agent()
        track_before = [p.detach().clone() for p in agent._track_parameters]
        hand_before = agent.model.mu_c.weight.detach().clone()
        norm_before = copy.deepcopy(agent.running_mean_std_track.state_dict())
        for group in agent.optimizer.param_groups:
            group['lr'] = agent.last_lr
        self.optimize(agent)
        for before, parameter in zip(track_before, agent._track_parameters):
            torch.testing.assert_close(before, parameter, rtol=0, atol=0)
        self.assertFalse(torch.equal(hand_before, agent.model.mu_c.weight))
        self.assertFalse(agent.running_mean_std_track.training)
        self.assertTrue(agent.running_mean_std_hand.training)
        for key, before in norm_before.items():
            torch.testing.assert_close(before, agent.running_mean_std_track.state_dict()[key])
        self.assertEqual(agent.optimizer.param_groups[0]['lr'], 0.)

        agent.agent_steps = 50
        for group in agent.optimizer.param_groups:
            group['lr'] = agent.last_lr
        self.optimize(agent)
        self.assertTrue(all(p.requires_grad for p in agent._track_parameters))
        self.assertFalse(torch.equal(track_before[0], agent._track_parameters[0]))
        self.assertAlmostEqual(agent.optimizer.param_groups[0]['lr'], agent.last_lr*.1)
        self.assertEqual(agent.optimizer.param_groups[1]['lr'], agent.last_lr)

    def test_checkpoint_restores_optimizer_progress_and_freeze_stage(self):
        first = self.agent()
        self.optimize(first)
        first.agent_steps, first.epoch_num = 42, 3
        first.last_lr, first.best_rewards = .0003, 12.
        first.save(str(self.path/'hold'))
        loaded = self.agent(self.config(track=False), 'restored')
        loaded.restore_train(str(self.path/'hold.pth'))
        self.assertEqual(loaded.agent_steps, 42)
        self.assertEqual(loaded.epoch_num, 3)
        self.assertEqual(loaded.best_rewards, 12.)
        self.assertTrue(loaded.track_frozen)
        self.assertEqual(len(loaded.optimizer.state), len(first.optimizer.state))
        for key, val in first.model.state_dict().items():
            torch.testing.assert_close(loaded.model.state_dict()[key], val)
        for group in loaded.optimizer.param_groups:
            self.assertAlmostEqual(group['lr'], 0 if group['branch']=='track' else .0003)
        # With the same next minibatch, restored Adam moments must produce the
        # same actual update, rather than merely having the same state count.
        next_rng = torch.get_rng_state()
        self.optimize(first)
        torch.set_rng_state(next_rng)
        self.optimize(loaded)
        for key, value in first.model.state_dict().items():
            torch.testing.assert_close(loaded.model.state_dict()[key], value, rtol=0, atol=0)
        for key, value in first.running_mean_std_hand.state_dict().items():
            torch.testing.assert_close(loaded.running_mean_std_hand.state_dict()[key], value,
                                       rtol=0, atol=0)
        loaded.agent_steps = 60
        loaded.set_train()
        loaded.save(str(self.path/'thawed'))
        thawed = self.agent(self.config(track=False), 'thawed_agent')
        thawed.restore_train(str(self.path/'thawed.pth'))
        self.assertFalse(thawed.track_frozen)
        self.assertAlmostEqual(thawed.optimizer.param_groups[0]['lr'], .00003)

    def test_training_loop_keeps_restored_progress(self):
        agent = self.agent()
        agent.agent_steps = 92
        agent.epoch_num = 6
        # Exercise the real new training loop without a simulator or PPO loss.
        observed = []
        def epoch():
            observed.append(agent.agent_steps)
            agent.agent_steps += agent.batch_size
            return ([torch.tensor(0.)],)*5
        with patch.object(agent, 'train_epoch', side_effect=epoch), \
             patch.object(agent, 'write_stats'), patch.object(agent, 'save'), \
             patch('gym_dcmm.algs.ppo_dcmm_5fe75d5f.ppo_dcmm_catch_hold.wandb.log'):
            agent.train()
        self.assertEqual(observed, [92])
        self.assertEqual(agent.agent_steps, 100)
        self.assertEqual(agent.epoch_num, 7)

    def test_mismatched_resume_options_rejected(self):
        agent = self.agent()
        agent.save(str(self.path/'hold'))
        cfg = self.config(track=False)
        cfg.bounce_grasp_training.track_lr_scale = .2
        loaded = self.agent(cfg, 'mismatch')
        with self.assertRaises(ValueError):
            loaded.restore_train(str(self.path/'hold.pth'))
        for bad in (dict(freeze_track_steps=-1), dict(track_lr_scale=2),
                    dict(freeze_track_steps=.5), dict(unknown=1)):
            with self.assertRaises(ValueError):
                training_options(bad)


if __name__ == '__main__':
    unittest.main()
