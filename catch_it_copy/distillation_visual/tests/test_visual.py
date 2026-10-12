import copy
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from distillation_visual import run
from distillation_visual import environments
from distillation_visual.core import (Student, EpisodeReplay, encode_depth, proprioception,
                                      sequence_loss, SCHEMA)


def config(stage='catch'):
    c = json.loads((run.ROOT / 'distillation_visual/configs' / (stage+'.json')).read_text())
    c.update(device='cpu', iterations=2, num_envs_per_task=2, rollout_steps=3,
             updates_per_iteration=1, batch_episodes_per_task=1, capacity_frames_per_task=32,
             calibration_steps=2, max_episode_steps=8, bptt_steps=2,
             baseline_episodes=1, eval_episodes=1, eval_every=1, save_every=1, max_eval_steps=20)
    c['vision'].update(height=64, width=64)
    c['model'].update(embedding=8, gru_hidden=12, mlp_hidden=[16])
    for task in c['tasks'].values():
        task['success_metric'] = 'success'
    return c


def observation(n=2):
    return {'base': {'v_lin_2d': np.zeros((n, 2), np.float32)},
            'arm': {'ee_pos3d': np.ones((n, 3), np.float32),
                    'ee_quat': np.ones((n, 4), np.float32),
                    'ee_v_lin_3d': np.ones((n, 3), np.float32)},
            'object': {'pos3d': np.ones((n, 3), np.float32), 'v_lin_3d': np.ones((n, 3), np.float32)},
            'hand': np.zeros((n, 12), np.float32)}


class FakePool:
    """Deterministic episodes with staggered terminations and fresh reset images."""
    def __init__(self, task, cfg, count, seed, viewer=False):
        self.count, self.cfg = count, cfg
        self.length = np.arange(count) + 2
        self.steps = np.zeros(count, dtype=int)
        self.obs = self.packet()

    def packet(self):
        v = self.cfg['vision']
        depth = np.broadcast_to(self.steps[:, None, None, None] / 10,
                                 (self.count, 1, v['height'], v['width'])).astype(np.float16).copy()
        return {'teacher_obs': observation(self.count), 'depth': depth}

    def step(self, actions):
        self.steps += 1
        done = self.steps >= self.length
        self.steps[done] = 0
        self.obs = self.packet()
        info = [{'success': True, '_truncated': bool(d), '_terminated': False,
                 'terminated_reason': 'test_success'} for d in done]
        return self.obs, np.ones(self.count), done, info

    def close(self):
        pass


class FakeTeacher:
    hash = 'fake'

    def __init__(self, action_dim):
        self.action_dim = action_dim

    def predict(self, raw):
        return np.full((len(raw), self.action_dim), .25, np.float32)


class VisualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_student_input_has_no_object_or_task_dependency(self):
        obs = observation()
        for stage, expected in [('track', 12), ('catch', 24)]:
            before = proprioception(obs, stage)
            changed = copy.deepcopy(obs)
            changed['object'] = {'pos3d': np.full((2, 3), np.nan)}
            changed['task'] = 'anything'
            np.testing.assert_array_equal(before, proprioception(changed, stage))
            self.assertEqual(before.shape, (2, expected))

    def test_metric_depth_is_fixed_scale(self):
        a = encode_depth(np.array([[.1, 1.1, 2.1, 9.]], np.float32), .1, 2.1)
        np.testing.assert_array_equal(a, np.array([[[0., .5, 1., 1.]]], np.float16))
        with self.assertRaises(ValueError):
            encode_depth(np.full((2, 2), np.nan), .1, 5.)

    def test_recurrent_sequence_matches_stepwise_and_reset(self):
        c = config()
        s = Student(c['stage'], c['vision'], c['model']).eval()
        depth = torch.rand(2, 4, 1, 64, 64)
        prop = torch.rand(2, 4, 24)
        previous = torch.rand(2, 4, 20)
        whole, _, _ = s(depth, prop, previous)
        h, steps = None, []
        for t in range(4):
            action, h, _ = s(depth[:, t:t+1], prop[:, t:t+1], previous[:, t:t+1], h)
            steps.append(action)
        torch.testing.assert_close(whole, torch.cat(steps, 1))
        reset_action, _, _ = s(depth[:, :1], prop[:, :1], previous[:, :1], torch.zeros_like(h))
        torch.testing.assert_close(reset_action, whole[:, :1])

    def test_padding_has_no_loss_or_gradient(self):
        logits = torch.randn(2, 4, 20, requires_grad=True)
        target = torch.zeros_like(logits)
        valid = torch.tensor([[1., 1., 0., 0.], [1., 1., 1., 1.]])
        loss = sequence_loss(logits, target, valid)
        loss.backward()
        self.assertEqual(logits.grad[0, 2:].abs().sum().item(), 0)
        changed = logits.detach().clone()
        changed[0, 2:] = 100
        torch.testing.assert_close(loss, sequence_loss(changed, target, valid))
        parts = sum(sequence_loss(logits[:, a:a+2], target[:, a:a+2], valid[:, a:a+2],
                                  episode_lengths=valid.sum(1)) for a in (0, 2))
        torch.testing.assert_close(loss, parts)

    def test_bce_is_stable_at_saturation(self):
        logits = torch.full((1, 2, 8), 100., requires_grad=True)
        loss = sequence_loss(logits, -torch.ones_like(logits), torch.ones(1, 2))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_replay_evicts_whole_episodes_and_roundtrips(self):
        replay = EpisodeReplay(['throw'], 5)
        for length in (2, 3, 4):
            rows = [{'depth': np.zeros((1, 64, 64), np.float16), 'proprio': np.zeros(12, np.float32),
                     'previous': np.zeros(8, np.float32), 'target': np.ones(8, np.float32)} for _ in range(length)]
            replay.add('throw', rows)
        self.assertEqual(replay.sizes['throw'], 4)
        restored = EpisodeReplay(['throw'], 5)
        restored.load_state_dict(replay.state_dict())
        batch = restored.sample(2, 'cpu')
        self.assertEqual(batch['valid'].sum().item(), 8)

    def test_task_specific_control_scales_rejected(self):
        c = config()
        c['tasks']['roll']['action_scales'][0] *= 2
        with self.assertRaisesRegex(ValueError, 'same physical scales'):
            run.validate(c)

    def test_camera_export_has_images_and_episode_boundaries(self):
        c = config('track')
        teachers = {t: FakeTeacher(8) for t in c['tasks']}
        with tempfile.TemporaryDirectory() as directory, patch.object(run, 'Pool', FakePool):
            run.camera_check(c, teachers, run.validate(c), Path(directory), 4)
            with np.load(Path(directory) / 'throw_depth.npz') as data:
                np.testing.assert_array_equal(data['done'], [False, True, False, True])
                self.assertEqual(data['depth_normalized'].shape, (4, 1, 64, 64))
            self.assertEqual((Path(directory) / 'throw_depth.png').read_bytes()[:8], b'\x89PNG\r\n\x1a\n')

    def test_worker_pairs_reset_depth_with_reset_state(self):
        class Environment:
            act_t_dim, obs_t_dim = 8, 18

            def __init__(self, **kwargs):
                self.episode = 0
                self.step_touch = False

            def reset(self):
                self.episode += 1
                self.render()  # Must be the disabled legacy renderer, not camera.
                obs = observation(1)
                obs = {k: {a: b[0] for a, b in v.items()} if isinstance(v, dict) else v[0]
                       for k, v in obs.items()}
                obs['object']['pos3d'][:] = self.episode
                return obs, {}

            def step(self, action):
                self.step_touch = True
                return None, 1., False, True, {'success': True}

            def close(self):
                pass

        class Camera:
            def __init__(self, env, config):
                self.env = env

            def capture(self):
                return np.full((1, 64, 64), self.env.episode / 10, np.float16)

            def close(self):
                pass

        class Pipe:
            def __init__(self):
                self.commands = iter([('step', {}), ('close', None)])
                self.messages = []

            def recv(self):
                return next(self.commands)

            def send(self, message):
                self.messages.append(message)

            def close(self):
                pass

        c = run.env_config(config('track'), 'throw')
        c['env_factory'] = 'fake_visual_worker:Environment'
        pipe = Pipe()
        module = types.ModuleType('fake_visual_worker')
        module.Environment = Environment
        with patch.dict('sys.modules', {'fake_visual_worker': module}), \
                patch.object(environments, 'DepthCamera', Camera), patch.dict('os.environ', {}):
            environments._worker(pipe, 'throw', c, 1, False)
        self.assertTrue(all(status == 'ok' for status, _ in pipe.messages))
        initial = pipe.messages[0][1]
        next_packet, _, done, info = pipe.messages[1][1]
        self.assertTrue(done)
        self.assertTrue(info['track_contact_success'])
        self.assertEqual(initial['teacher_obs']['object']['pos3d'][0], 1)
        self.assertEqual(next_packet['teacher_obs']['object']['pos3d'][0], 2)
        self.assertAlmostEqual(float(next_packet['depth'][0, 0, 0]), .2, places=3)

    def test_collector_reset_and_hidden_refresh(self):
        c = config('track')
        student = run.make_student(c)
        teachers = {t: FakeTeacher(8) for t in c['tasks']}
        replay = EpisodeReplay(c['tasks'], 32)
        with patch.object(run, 'Pool', FakePool):
            collector = run.Collector(c, student, teachers, run.validate(c), replay)
            try:
                collector.collect(3)
                # Worker 0 has one step in its second episode; worker 1 just reset.
                self.assertEqual(len(collector.pending['throw'][0]), 1)
                self.assertEqual(len(collector.pending['throw'][1]), 0)
                self.assertEqual(collector.hidden['throw'][:, 1].abs().sum().item(), 0)
                self.assertEqual(np.abs(collector.previous['throw'][1]).sum(), 0)
                with torch.no_grad():
                    student.gru.weight_ih_l0.add_(.01)
                collector.refresh_hidden()
                row = collector.pending['throw'][0][0]
                values = [torch.as_tensor(row[k])[None, None] for k in ('depth', 'proprio', 'previous')]
                _, expected, _ = student(*values)
                torch.testing.assert_close(collector.hidden['throw'][:, :1], expected)
            finally:
                collector.close()

    def test_fake_train_evaluate_resume_both_stages(self):
        for stage in ('track', 'catch'):
            c = config(stage)
            teachers = {t: FakeTeacher(8 if stage == 'track' else 20) for t in c['tasks']}
            adapters = run.validate(c)
            with tempfile.TemporaryDirectory() as directory, patch.object(run, 'Pool', FakePool), \
                    patch.object(run, 'provenance', return_value={'test': 'stable'}):
                output = Path(directory)
                run.train(c, teachers, adapters, output)
                saved = torch.load(output / 'student_last.pth', weights_only=True)
                self.assertEqual(saved['schema'], SCHEMA)
                self.assertEqual(saved['iteration'], 2)
                student = run.make_student(c)
                student.load_state_dict(saved['student'])
                result = run.evaluate(c, adapters, student=student)
                self.assertTrue(all(r['success_rate'] == 1 for r in result.values()))
                c['iterations'] = 3
                run.train(c, teachers, adapters, output, output / 'resume_last.pth')
                saved = torch.load(output / 'resume_last.pth', weights_only=True)
                self.assertEqual(saved['iteration'], 3)
                self.assertTrue(saved['student']['normalizer_fitted'])


if __name__ == '__main__':
    unittest.main()
