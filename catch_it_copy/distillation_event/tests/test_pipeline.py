"""Synthetic pipeline verification, not robot training or success-rate evidence."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from distillation_visual_aux.tests.test_auxiliary import FakePool as OriginalPool
from distillation_visual.tests.test_visual import config as small_config
from distillation_event import run


class Pool(OriginalPool):
    def step(self, actions):
        obs, reward, done, infos = super().step(actions)
        for info in infos:
            info['distillation_events'] = np.array([0, 0, 1, 0], np.float32)
        return obs, reward, done, infos


class Teacher:
    hash = 'synthetic'
    def __init__(self, stage):
        self.stage = stage
        self.model = nn.Module()
        if stage == 'track':
            self.model.actor_mlp = nn.Linear(18, 8)
        else:
            self.model.actor_mlp_t = nn.Linear(18, 8)
            self.model.actor_mlp_c = nn.Linear(28, 8)
        self.model.eval().requires_grad_(False)

    def predict(self, raw):
        x = torch.as_tensor(raw)
        if self.stage == 'track':
            self.model.actor_mlp(x)
        else:
            self.model.actor_mlp_t(x[:, :18])
            self.model.actor_mlp_c(x[:, 2:])
        return np.full((len(raw), 8 if self.stage == 'track' else 20), .25, np.float32)


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def configuration(self, stage):
        c = json.loads((run.ROOT / 'distillation_event/configs' / (stage + '.json')).read_text())
        for key, value in small_config(stage).items():
            if key not in ('model', 'auxiliary', 'events'):
                c[key] = value
        c['model'].update(embedding=8, gru_hidden=12, mlp_hidden=[16])
        return c

    def test_training_checkpoint_resume_and_history(self):
        for stage in ('track', 'catch'):
            c = self.configuration(stage)
            adapters = run.validate(c)
            teachers = {t: Teacher(stage) for t in c['tasks']}
            with tempfile.TemporaryDirectory() as d, patch.object(run, 'Pool', Pool), patch.object(run, 'provenance', return_value={'synthetic': '1'}):
                output = Path(d)
                run.train(c, teachers, adapters, output)
                checkpoint = torch.load(output / 'student_last.pth', weights_only=True)
                student = run.make_student(c)
                student.load_state_dict(checkpoint['student'])
                depth = torch.rand(1, 3, 2, 64, 64)
                state = torch.randn(1, 3, student.proprio_dim)
                previous = torch.randn(1, 3, student.action_dim)
                full = student(depth, state, previous)[0]
                hidden, parts = None, []
                for i in range(3):
                    action, hidden, _ = student(depth[:, i:i+1], state[:, i:i+1], previous[:, i:i+1], hidden)
                    parts.append(action)
                torch.testing.assert_close(full, torch.cat(parts, 1), atol=1e-6, rtol=1e-5)
                self.assertTrue(torch.isfinite(full).all())
                self.assertLessEqual(full.abs().max(), 1.)
                c['iterations'] = 3
                run.train(c, teachers, adapters, output, output / 'resume_last.pth')
                self.assertEqual(torch.load(output / 'resume_last.pth', weights_only=True)['iteration'], 3)
                self.assertTrue(all(p.grad is None for teacher in teachers.values() for p in teacher.model.parameters()))


if __name__ == '__main__':
    unittest.main()
