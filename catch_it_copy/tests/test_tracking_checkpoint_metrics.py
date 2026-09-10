"""CPU regressions for real Tracking train/checkpoint code without MuJoCo."""
import ast
import contextlib
import importlib.util
import io
from itertools import count
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PPO_DIR = ROOT / 'gym_dcmm/algs/ppo_dcmm'
TRACK_TREE = ast.parse((PPO_DIR / 'ppo_dcmm_track.py').read_text(encoding='utf-8'))
TRACK_CLASS = next(node for node in TRACK_TREE.body
                   if isinstance(node, ast.ClassDef) and node.name == 'PPO_Track')


def actual_method(name, scope):
    node = next(node for node in TRACK_CLASS.body
                if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<Tracking method>', 'exec'), scope)
    return scope[name]


def meter_class():
    tree = ast.parse((PPO_DIR / 'utils.py').read_text(encoding='utf-8'))
    node = next(node for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == 'AverageScalarMeter')
    scope = dict(torch=torch, np=np)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual meter>', 'exec'), scope)
    return scope['AverageScalarMeter']


def policy(actions, obs=21):
    spec = importlib.util.spec_from_file_location('tracking_policy', PPO_DIR / 'models_track.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ActorCritic(dict(actor_units=[16, 16], actions_num=actions,
                                  input_shape=(obs,), separate_value_mlp=True))


class CheckpointTests(unittest.TestCase):
    def test_empty_rollout_does_not_block_negative_best_or_log_fake_zero(self):
        meter = meter_class()
        tb_logs, wandb_logs, saves = [], [], []
        agent = SimpleNamespace(
            env=SimpleNamespace(reset=lambda: ({}, {}), call=lambda _: ['throw_basket']),
            obs2tensor=lambda obs: obs, batch_size=4, max_agent_steps=16, epoch_num=0,
            storage=SimpleNamespace(data_dict={}), lr_schedule='constant',
            data_collect_time=0, rl_train_time=0, nn_dir='unused', save_freq=1,
            write_stats=lambda *args: None,
            writer=SimpleNamespace(add_scalar=lambda *args: tb_logs.append(args)),
            episode_rewards=meter(200), episode_lengths=meter(200), episode_success=meter(200))
        # Execute the actual initializer assignment as well as the full train
        # loop, so even a first valid return below -10000 can be selected.
        initializer = next(node for node in TRACK_CLASS.body
                           if isinstance(node, ast.FunctionDef) and node.name == '__init__')
        best_assignment = next(node for node in initializer.body if isinstance(node, ast.Assign)
                               and any(isinstance(target, ast.Attribute) and target.attr == 'best_rewards'
                                       for target in node.targets))
        exec(compile(ast.Module(body=[best_assignment], type_ignores=[]), '<best init>', 'exec'),
             dict(self=agent))

        def train_epoch():
            agent.agent_steps += agent.batch_size
            if agent.epoch_num >= 2:
                reward = -15000. if agent.epoch_num == 2 else -100.
                agent.episode_rewards.update(torch.tensor([reward]))
                agent.episode_lengths.update(torch.tensor([100.]))
                agent.episode_success.update(torch.tensor([0.]))
            return [], [], [], [], []

        agent.train_epoch = train_epoch
        agent.save = lambda path: saves.append((agent.epoch_num, Path(path).name))
        ticks = count()
        train = actual_method('train', dict(time=SimpleNamespace(time=lambda: float(next(ticks))), os=os,
                              wandb=SimpleNamespace(log=lambda data, step: wandb_logs.append((step, data)))))
        with contextlib.redirect_stdout(io.StringIO()):
            train(agent)

        bests = [(epoch, name) for epoch, name in saves if '_best_reward_' in name]
        self.assertEqual(bests, [(2, 'throw_basket_track_best_reward_-15000.00'),
                                 (3, 'throw_basket_track_best_reward_-7550.00')])
        self.assertEqual(agent.best_rewards, -7550.)
        self.assertEqual([epoch for epoch, name in saves if name.endswith('_last')], [1, 2, 3])
        self.assertTrue(any(epoch == 1 and name.endswith('_unscored') for epoch, name in saves))
        self.assertEqual([value for name, value, step in tb_logs
                          if name == 'metrics/episode_rewards_per_step'], [-15000., -7550.])
        self.assertEqual([step for step, data in wandb_logs], [12, 16])
        self.assertTrue(all(step >= 12 for name, value, step in tb_logs))

    def test_base_only_rejects_old_action_and_observation_shapes_without_mutation(self):
        target = policy(2)
        before = {name: value.clone() for name, value in target.state_dict().items()}
        agent = SimpleNamespace(actions_num=2, model=target)
        load = actual_method('_load_compatible_model_state', {})
        old_states = [policy(8).state_dict(), policy(2, obs=18).state_dict()]
        missing = dict(before)
        missing.pop('mu.weight')
        old_states.append(missing)
        for state in old_states:
            with self.subTest(keys=list(state)):
                with self.assertRaisesRegex(ValueError, 'matching 2-action checkpoint'):
                    load(agent, state)
                for name, value in target.state_dict().items():
                    torch.testing.assert_close(value, before[name])

    def test_matching_base_only_checkpoint_loads_all_tensors(self):
        source, target = policy(2), policy(2)
        agent = SimpleNamespace(actions_num=2, model=target)
        load = actual_method('_load_compatible_model_state', {})
        load(agent, source.state_dict())
        for name, value in target.state_dict().items():
            torch.testing.assert_close(value, source.state_dict()[name])

    def test_other_tracking_modes_keep_existing_partial_loading(self):
        source, target = policy(6, obs=18), policy(8, obs=18)
        extra_rows = target.mu.weight[6:].detach().clone()
        agent = SimpleNamespace(actions_num=8, model=target)
        load = actual_method('_load_compatible_model_state', {})
        with contextlib.redirect_stdout(io.StringIO()):
            load(agent, source.state_dict())
        torch.testing.assert_close(target.mu.weight[:6], source.mu.weight)
        torch.testing.assert_close(target.mu.weight[6:], extra_rows)


if __name__ == '__main__':
    unittest.main()
