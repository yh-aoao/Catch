"""CPU-only checks for vector evaluation accounting and per-env memory resets."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import evaluate_aux_parallel as evaluation

spec = importlib.util.spec_from_file_location('evaluation_metrics',
    evaluation.ROOT/'catch_it_copy/distillation/metrics.py')
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


def stack_tree(items):
    if isinstance(items[0], dict):
        return {key: stack_tree([item[key] for item in items]) for key in items[0]}
    return np.stack(items)


class FakePool:
    latest = None

    def __init__(self, task, config, count, seed):
        FakePool.latest = self
        self.steps, self.episodes = [0]*count, [0]*count
        self.pending, self.closed = {}, False
        self.pipes = [SimpleNamespace(send=lambda payload, i=i: self.send(i, payload)) for i in range(count)]
        self.obs = stack_tree([self.packet(i, True) for i in range(count)])

    def packet(self, i, reset):
        obs = dict(base={'v_lin_2d': np.array([i, self.steps[i]], np.float32)},
                   arm={}, hand=np.zeros(12, np.float32))
        if reset:
            obs['_evaluation_initial_json'] = json.dumps({'physical_hash': '{}:{}'.format(i, self.episodes[i])})
        return {'teacher_obs': obs, 'depth': np.full((2, 2, 2), i, np.float16)}

    def send(self, i, payload):
        assert payload[0] == 'step' and i not in self.pending
        self.pending[i] = payload[1]

    def _receive(self, pipe):
        i = self.pipes.index(pipe)
        assert i in self.pending
        del self.pending[i]
        self.steps[i] += 1
        done = self.steps[i] == i % 3+1
        info = dict(success=(i+self.episodes[i]) % 2 == 0,
                    _truncated=done, terminated_reason='synthetic', success_version='synthetic')
        if done:
            self.episodes[i] += 1
            self.steps[i] = 0
        return self.packet(i, done), float(i+1), done, info

    def close(self):
        self.closed = True


class FakeStudent:
    action_dim = 2

    def initial_state(self, count):
        return torch.zeros(1, count, 1)

    def __call__(self, depth, proprio, previous, hidden):
        # The current step is an independent observable oracle for each worker.
        ticks = proprio[:, 0, 1]
        torch.testing.assert_close(hidden[0, :, 0], ticks)
        torch.testing.assert_close(previous[:, 0], ticks[:, None].expand(-1, 2))
        state = hidden+1
        action = state.transpose(0, 1).expand(-1, -1, 2)
        return action, state, action


class VectorEvaluationTest(unittest.TestCase):
    def evaluate(self, output, environments, episodes, batch=0, max_steps=8):
        run = SimpleNamespace(TASK_IDS={'throw': 0}, env_config=lambda config, task: {}, Pool=FakePool,
                              proprioception=lambda obs, stage: obs['base']['v_lin_2d'],
                              terminal_success=lambda info, metric: info[metric], wilson=lambda n, total: [0., 1.])
        config = {'stage': 'catch', 'max_episode_steps': max_steps,
                  'tasks': {'throw': {'success_metric': 'success', 'success_definition': 'synthetic'}},
                  'vision': {}}
        adapter = SimpleNamespace(executable=lambda command: command,
                                  action_dict=lambda command: {'action': command})
        args = SimpleNamespace(episodes=episodes, num_envs=environments, seed=20261029,
                               device='cpu', inference_batch_size=batch, worker_timeout=10)
        with contextlib.redirect_stdout(io.StringIO()):
            return evaluation.evaluate_task(run, config, FakeStudent(), adapter, args,
                                            'throw', output, {}, metrics, stack_tree)

    def test_quotas_and_no_fast_worker_selection(self):
        self.assertEqual(evaluation.episode_quotas(300, 32), [10]*12+[9]*20)
        for batch in [0, 1, 2]:
            with self.subTest(batch=batch), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp)/'task'
                result = self.evaluate(output, 4, 11, batch)
                rows = result['records']
                self.assertEqual([row['episode_id'] for row in rows], list(range(11)))
                self.assertEqual([sum(row['worker_id'] == i for row in rows) for i in range(4)], [3,3,3,2])
                self.assertEqual([row['length'] for row in rows], [i % 4 % 3+1 for i in range(11)])
                self.assertEqual(result['successes'], 6)
                for row in rows:
                    self.assertEqual(row['initial']['physical_hash'], '{}:{}'.format(row['worker_id'], row['worker_episode_id']))
                    self.assertEqual(row['reset_seed'], 21261029+row['worker_id']*1009+row['worker_episode_id']*100003)
                self.assertTrue((output/'COMPLETE').exists())
                self.assertTrue(FakePool.latest.closed)

    def test_less_episodes_than_environments(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.evaluate(Path(tmp)/'task', 32, 3)
            self.assertEqual(len(FakePool.latest.pipes), 3)
            self.assertEqual(result['episodes'], 3)
            self.assertEqual(result['successes'], 2)

    def test_worker_cleanup_when_episode_never_ends_in_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'task'
            with self.assertRaisesRegex(RuntimeError, 'max_episode_steps'):
                self.evaluate(output, 2, 4, max_steps=1)
            self.assertTrue(FakePool.latest.closed)
            self.assertFalse((output/'COMPLETE').exists())

    def test_restore_hashes_and_refuse_to_overwrite_modified_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            contents = b'print("historical")\n'
            with tarfile.open(base/'historical_sources_and_assets.tar.gz', 'w:gz') as archive:
                member = tarfile.TarInfo('future/module.py')
                member.size = len(contents)
                archive.addfile(member, io.BytesIO(contents))
            audit = {'source_root': str(base/'restored/future'),
                     'files': [{'path': 'module.py', 'actual': hashlib.sha256(contents).hexdigest()}]}
            with patch.object(evaluation, 'PRIOR', base), contextlib.redirect_stdout(io.StringIO()):
                source = evaluation.restore_and_verify(audit, 'future/')
                self.assertEqual((source/'module.py').read_bytes(), contents)
                (source/'module.py').write_text('user modification')
                with self.assertRaisesRegex(RuntimeError, 'Audited source changed'):
                    evaluation.restore_and_verify(audit, 'future/')
                self.assertEqual((source/'module.py').read_text(), 'user modification')


if __name__ == '__main__':
    unittest.main()
