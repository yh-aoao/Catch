import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from distillation.core import ActionAdapter, Replay, Student, imitation_loss, student_input
from distillation.environments import Pool, terminal_success
from distillation.run import train, evaluate, make_adapters
from distillation.teachers import ROOT, Teacher


def observation(n):
    return {'base': {'v_lin_2d': np.zeros((n, 2), np.float32)},
            'arm': {'ee_pos3d': np.ones((n, 3), np.float32),
                    'ee_quat': np.ones((n, 4), np.float32),
                    'ee_v_lin_3d': np.zeros((n, 3), np.float32)},
            'object': {'pos3d': np.ones((n, 3), np.float32),
                       'v_lin_3d': np.zeros((n, 3), np.float32)},
            'hand': np.zeros((n, 12), np.float32)}


class FakePool:
    """Two-step episodes with autoreset observation, distinct from terminal info."""
    def __init__(self, task, config, count, seed, viewer=False):
        self.n, self.steps = count, 0
        self.obs = observation(count)
    def step(self, actions):
        self.steps += 1
        done = np.full(self.n, self.steps % 2 == 0)
        infos = [{'success': True, 'roll_eval_success': True,
                  '_truncated': bool(d), '_terminated': False} for d in done]
        return self.obs, np.ones(self.n), done, infos
    def close(self):
        pass


class FakeTeacher:
    hash = 'fake_teacher'
    def predict(self, raw):
        return np.full((len(raw), 20), .2, np.float32)


class DistillationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = json.loads((ROOT / 'distillation/configs/three_tasks.json').read_text())

    def test_unknown_mapping_rejected_and_hand_not_shifted(self):
        unresolved = dict(self.config['tasks']['throw'], action_indices=None)
        with self.assertRaisesRegex(ValueError, 'Unresolved'):
            ActionAdapter(unresolved)
        cfg = copy.deepcopy(self.config['tasks']['roll'])
        # Synthetic mapping tests the mechanism, NOT a claim about real throw axes.
        cfg['action_indices'] = [0, 1, 2, 3, 4, 7] + list(range(8, 20))
        adapter = ActionAdapter(cfg)
        native = np.arange(18, dtype=np.float32)[None] / 20
        command = adapter.canonical(native)
        np.testing.assert_equal(command[0, 8:], native[0, 6:])
        np.testing.assert_equal(command[0, 5:7], 0)
        command[0, 5:7] = .9
        np.testing.assert_equal(adapter.executable(command)[0, 5:7], 0)

    def test_group_loss_and_mask(self):
        pred = torch.zeros(2, 20, requires_grad=True)
        target = torch.ones(2, 20)
        mask = torch.ones(2, 20); mask[:, 5:7] = 0
        loss = imitation_loss(pred, target, mask)
        self.assertAlmostEqual(loss.item(), 1.)
        loss.backward()
        self.assertEqual(pred.grad[:, 5:7].abs().sum().item(), 0)
        self.assertAlmostEqual(pred.grad[:, :2].abs().sum().item(),
                               pred.grad[:, 8:].abs().sum().item(), places=6)

    def test_balanced_ring_restore(self):
        replay = Replay(['roll', 'bounce'], 5)
        for task, number in [('roll', 1), ('bounce', 2)]:
            replay.add(task, np.full((9, 53), number, np.float32),
                       np.zeros((9, 20), np.float32), np.ones(20))
        restored = Replay(['roll', 'bounce'], 5)
        restored.load_state_dict(replay.state_dict())
        x, _, _ = restored.sample(7, 'cpu')
        self.assertEqual(int((x[:, 0] == 1).sum()), 7)
        self.assertEqual(int((x[:, 0] == 2).sum()), 7)
        self.assertEqual(restored.data['roll']['pos'], 4)

    def test_normalizer_frozen(self):
        student = Student([8])
        x = torch.randn(20, 53)
        student.fit_normalizer(x)
        before = student.mean.clone()
        student.train(); student(x * 10)
        torch.testing.assert_close(before, student.mean)
        with self.assertRaises(RuntimeError):
            student.fit_normalizer(x)

    def test_terminal_success_not_timeout(self):
        info = {'success': False, '_truncated': True}
        self.assertFalse(terminal_success(info, 'success'))
        self.assertTrue(terminal_success(info, 'legacy_truncated'))
        with self.assertRaises(ValueError):
            terminal_success(info, 'missing')

    def test_spawn_reset_and_error_propagation(self):
        config = {'env_factory': 'distillation.tests.fake_environment:Environment',
                  'env_kwargs': {}, 'worker_timeout_seconds': 30}
        pool = Pool('roll', config, 1, 42)
        try:
            obs, _, done, info = pool.step({'base': np.zeros((1, 2))})
            self.assertTrue(done[0])
            self.assertTrue(info[0]['success'])
            self.assertEqual(obs['episode'][0, 0], 2)
            with self.assertRaisesRegex(RuntimeError, 'intentional worker error'):
                pool.step({'base': -np.ones((1, 2))})
        finally:
            pool.close()

    def test_real_teacher_strict_and_frozen(self):
        for task in ('roll', 'bounce'):
            cfg = self.config['tasks'][task]
            teacher = Teacher(cfg)
            before = {k: v.clone() for k, v in teacher.track.state_dict().items()}
            self.assertLessEqual(teacher.parity_check(np.zeros((4, 30), np.float32)), 1e-5)
            self.assertTrue(all(not p.requires_grad for p in teacher.model.parameters()))
            for k, v in teacher.track.state_dict().items():
                torch.testing.assert_close(before[k], v)

    def test_new_throw_checkpoint_when_available(self):
        cfg = self.config['tasks']['throw']
        if not (ROOT / cfg['checkpoint']).is_file():
            self.skipTest('New throw checkpoint is server-only; actual weight validation must run there')
        teacher = Teacher(cfg)
        self.assertEqual(teacher.action_dim, 20)
        self.assertLessEqual(teacher.parity_check(np.zeros((4, 30), np.float32)), 1e-5)
        ActionAdapter(cfg).canonical(teacher.predict(np.zeros((4, 30), np.float32)))

    def test_throw_rejects_old_18d_checkpoint(self):
        cfg = dict(self.config['tasks']['throw'], checkpoint='best_model/catch_two_stage.pth', sha256=None)
        with self.assertRaisesRegex(ValueError, 'expected 20, got 18'):
            Teacher(cfg)

    def test_throw_20d_adapter_with_fixture(self):
        # Roll weights provide a 20D shape fixture only, not verification of server throw weights.
        cfg = dict(self.config['tasks']['throw'],
                   checkpoint=self.config['tasks']['roll']['checkpoint'], sha256=None)
        teacher = Teacher(cfg)
        self.assertEqual(teacher.action_dim, 20)
        self.assertEqual(ActionAdapter(cfg).indices, list(range(20)))
        self.assertEqual(teacher.parity_check(np.zeros((4, 30), np.float32)), 0.)

    def test_online_train_resume_and_student_only_eval(self):
        cfg = copy.deepcopy(self.config)
        cfg.update(device='cpu', hidden=[16], iterations=2, num_envs_per_task=2,
                   rollout_steps=2, calibration_steps=2, updates_per_iteration=2,
                   batch_per_task=4, capacity_per_task=32, eval_every=1,
                   baseline_episodes=2, eval_episodes=2, max_eval_steps=10)
        teachers = {t: FakeTeacher() for t in cfg['tasks']}
        adapters = make_adapters(cfg)
        with tempfile.TemporaryDirectory() as folder, patch('distillation.environments.Pool', FakePool):
            output = Path(folder)
            train(cfg, teachers, adapters, output)
            saved = torch.load(output / 'resume_last.pth', weights_only=True)
            # Steps alternate first/second episode states; prev command resets to zero.
            x = saved['replay']['roll']['x']
            np.testing.assert_equal(x[0:2, 30:50].numpy(), 0)
            np.testing.assert_equal(x[4:6, 30:50].numpy(), 0)
            self.assertGreater(x[2:4, 30:50].abs().sum().item(), 0)
            self.assertEqual(saved['iteration'], 2)
            cfg['iterations'] = 3
            train(cfg, teachers, adapters, output, output / 'resume_last.pth')
            package = torch.load(output / 'student_last.pth', weights_only=True)
            self.assertNotIn('replay', package)
            self.assertNotIn('optimizer', package)
            student = Student(cfg['hidden'])
            student.load_state_dict(package['student'])
            result = evaluate(cfg, adapters, student=student, teachers=None)
            self.assertEqual(result['roll']['episodes'], 2)
            self.assertEqual(package['iteration'], 3)


if __name__ == '__main__':
    unittest.main()
