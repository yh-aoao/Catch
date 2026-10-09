import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from distillation_track.core import ActionAdapter, Student, imitation_loss, student_input
from distillation_track.environments import Pool, terminal_success
from distillation_track.run import check, evaluate, make_adapters, make_teachers, train
from distillation_track.teachers import ROOT, Teacher


def observation(n):
    return {'base': {'v_lin_2d': np.zeros((n, 2), np.float32)},
            'arm': {'ee_pos3d': np.ones((n, 3), np.float32),
                    'ee_quat': np.ones((n, 4), np.float32),
                    'ee_v_lin_3d': np.zeros((n, 3), np.float32)},
            'object': {'pos3d': np.ones((n, 3), np.float32),
                       'v_lin_3d': np.zeros((n, 3), np.float32)}}


class FakePool:
    def __init__(self, task, config, count, seed, viewer=False):
        self.n, self.steps = count, 0
        self.obs = observation(count)

    def step(self, actions):
        assert actions['base'].shape == (self.n, 2)
        assert actions['arm'].shape == (self.n, 6)
        np.testing.assert_array_equal(actions['hand'], np.zeros((self.n, 12)))
        self.steps += 1
        done = np.full(self.n, self.steps % 2 == 0)
        infos = [{'success': True, 'roll_eval_success': True,
                  '_truncated': bool(d), '_terminated': False} for d in done]
        return self.obs, np.ones(self.n), done, infos

    def close(self):
        pass


class TrackingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = json.loads((ROOT / 'distillation_track/configs/three_tasks.json').read_text())

    def test_real_weights_and_original_inference(self):
        teachers = make_teachers(dict(self.config, device='cpu'))
        report = check(self.config, teachers)
        for task, teacher in teachers.items():
            self.assertEqual(report[task]['inference_max_error'], 0.)
            before = {k: v.clone() for k, v in teacher.rms.state_dict().items()}
            teacher.predict(np.random.randn(11, 18).astype(np.float32))
            for key, value in teacher.rms.state_dict().items():
                torch.testing.assert_close(value, before[key])
            self.assertFalse(any(p.requires_grad for p in teacher.model.parameters()))
        self.assertFalse(report['throw']['interface_ready'])
        for task in ('roll', 'bounce'):
            self.assertTrue(report[task]['action_parity'])
            self.assertTrue(report[task]['interface_ready'])

    def test_unresolved_throw_and_wrong_checkpoint_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unresolved'):
            ActionAdapter(self.config['tasks']['throw'])
        cfg = dict(self.config['tasks']['roll'], expected_action_dim=6)
        with self.assertRaisesRegex(ValueError, 'action dimension'):
            Teacher(cfg)
        cfg = dict(self.config['tasks']['roll'], sha256='wrong')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            Teacher(cfg)

    def test_input_action_mask_and_group_loss(self):
        x = student_input(observation(2), np.zeros((2, 8), np.float32), 'bounce')
        self.assertEqual(x.shape, (2, 29))
        np.testing.assert_array_equal(x[:, -3:], [[0, 0, 1]] * 2)
        # Synthetic mapping exercises masking; it does not establish throw semantics.
        adapter = ActionAdapter(dict(self.config['tasks']['throw'], action_indices=[0, 1, 2, 3, 4, 7]))
        y = adapter.canonical(np.full((2, 6), .5, np.float32))
        np.testing.assert_array_equal(y[:, 5:7], 0)
        np.testing.assert_array_equal(adapter.executable(np.ones((2, 8)))[:, 5:7], 0)
        pred = torch.zeros(2, 8, requires_grad=True)
        loss = imitation_loss(pred, torch.ones(2, 8), torch.tensor(adapter.mask).expand(2, -1))
        loss.backward()
        self.assertAlmostEqual(loss.item(), 1.)
        self.assertEqual(pred.grad[:, 5:7].abs().sum().item(), 0.)
        self.assertAlmostEqual(pred.grad[:, :2].abs().sum().item(), pred.grad[:, 2:].abs().sum().item())

    def test_metrics_match_selected_original_tests(self):
        self.assertEqual(self.config['tasks']['roll']['success_metric'], 'roll_eval_success')
        self.assertEqual(self.config['tasks']['bounce']['success_metric'], 'legacy_truncated')
        info = {'success': False, 'roll_eval_success': True, '_truncated': True}
        self.assertFalse(terminal_success(info, 'success'))
        self.assertTrue(terminal_success(info, 'roll_eval_success'))
        self.assertTrue(terminal_success(info, 'legacy_truncated'))

    def test_real_teachers_train_resume_and_student_only_evaluation(self):
        cfg = copy.deepcopy(self.config)
        cfg['tasks'].pop('throw')
        cfg.update(device='cpu', hidden=[16], iterations=2, num_envs_per_task=2,
                   rollout_steps=2, calibration_steps=2, updates_per_iteration=2,
                   batch_per_task=4, capacity_per_task=32, eval_every=1,
                   baseline_episodes=2, eval_episodes=2, max_eval_steps=10)
        teachers, adapters = make_teachers(cfg), make_adapters(cfg)
        with tempfile.TemporaryDirectory() as folder, patch('distillation_track.environments.Pool', FakePool):
            output = Path(folder)
            train(cfg, teachers, adapters, output)
            saved = torch.load(output / 'resume_last.pth', weights_only=True)
            x = saved['replay']['roll']['x']
            np.testing.assert_array_equal(x[:2, 18:26], 0)
            np.testing.assert_array_equal(x[4:6, 18:26], 0)
            self.assertGreater(x[2:4, 18:26].abs().sum().item(), 0)
            cfg['iterations'] = 3
            train(cfg, teachers, adapters, output, output / 'resume_last.pth')
            package = torch.load(output / 'student_last.pth', weights_only=True)
            self.assertEqual(package['iteration'], 3)
            self.assertNotIn('optimizer', package)
            self.assertNotIn('replay', package)
            student = Student(cfg['hidden'])
            student.load_state_dict(package['student'], strict=True)
            result = evaluate(cfg, adapters, student=student)
            self.assertEqual(result['roll']['episodes'], 2)
            self.assertTrue((output / 'teacher_baseline.episodes.jsonl').is_file())

    def test_spawn_tracking_mode_terminal_info_and_error(self):
        cfg = {'env_factory': 'distillation_track.tests.fake_environment:Environment',
               'env_kwargs': {}, 'worker_timeout_seconds': 30}
        pool = Pool('bounce', cfg, 1, 42)
        try:
            obs, _, done, info = pool.step({'base': np.zeros((1, 2))})
            self.assertTrue(done[0])
            self.assertTrue(info[0]['track_contact_success'])
            self.assertEqual(obs['episode'][0, 0], 2)
            with self.assertRaisesRegex(RuntimeError, 'intentional'):
                pool.step({'base': -np.ones((1, 2))})
        finally:
            pool.close()


if __name__ == '__main__':
    unittest.main()
