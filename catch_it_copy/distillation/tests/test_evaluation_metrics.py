import ast
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from distillation.run import evaluate, make_adapters, train
from distillation.teachers import ROOT
from distillation.tests.test_distillation import FakePool, FakeTeacher


class MissingMetricPool(FakePool):
    def __init__(self, task, *args, **kwargs):
        self.task = task
        super().__init__(task, *args, **kwargs)

    def step(self, actions):
        obs, reward, done, infos = super().step(actions)
        if self.task == 'bounce':
            for info in infos:
                info.pop('success')
        return obs, reward, done, infos


class ZeroPool(FakePool):
    def step(self, actions):
        obs, reward, done, infos = super().step(actions)
        for info in infos:
            info.update(success=False, roll_eval_success=False, _truncated=False)
        return obs, reward, done, infos


class EvaluationMetricsTests(unittest.TestCase):
    def config(self):
        cfg = json.loads((ROOT / 'distillation/configs/three_tasks.json').read_text())
        cfg['tasks'].pop('throw')
        cfg.update(device='cpu', hidden=[8], iterations=1, num_envs_per_task=1,
                   rollout_steps=1, calibration_steps=1, updates_per_iteration=1,
                   batch_per_task=1, capacity_per_task=8, eval_every=1,
                   baseline_episodes=1, eval_episodes=1, max_eval_steps=5)
        return cfg

    def test_real_bounce_info_always_has_false_default(self):
        # Execute the actual source method without importing MuJoCo.
        path = ROOT / 'gym_dcmm/envs/DcmmVecEnv.py'
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BounceEnv')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_get_info')
        ns = {'np': np}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), 'exec'), ns)
        body = SimpleNamespace(xpos=np.zeros(3))
        env = SimpleNamespace(Dcmm=SimpleNamespace(data=SimpleNamespace(time=1., body=lambda _: body),
                                                   object_name='object'), start_time=0., print_info=False)
        self.assertIs(ns['_get_info'](env)['success'], False)
        # Source still contains explicit success assignment; no timeout fallback introduced.
        step = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'step')
        self.assertTrue(any(isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
                            and n.value.value is True and any(isinstance(t, ast.Subscript)
                            and isinstance(t.value, ast.Name) and t.value.id == 'info'
                            for t in n.targets) for n in ast.walk(step)))

    def test_completed_task_survives_later_metric_error(self):
        cfg = self.config()
        cfg['tasks']['bounce']['success_metric'] = 'success'
        teachers = {t: FakeTeacher() for t in cfg['tasks']}
        with tempfile.TemporaryDirectory() as folder, patch('distillation.environments.Pool', MissingMetricPool):
            target = Path(folder) / 'baseline.json'
            with self.assertRaisesRegex(ValueError, 'Missing configured'):
                evaluate(cfg, make_adapters(cfg), teachers=teachers, output_file=target)
            result = json.loads(target.read_text())
            self.assertTrue(result['roll']['complete'])
            self.assertNotIn('bounce', result)
            records = [json.loads(line) for line in target.with_suffix('.episodes.jsonl').read_text().splitlines()]
            self.assertEqual(records[-1]['task'], 'bounce')
            self.assertNotIn('success', records[-1]['diagnostics'])

    def test_bounce_default_uses_original_timeout_metric(self):
        cfg = self.config()
        cfg['tasks'].pop('roll')
        self.assertEqual(cfg['tasks']['bounce']['success_metric'], 'legacy_truncated')
        # Legacy test counts truncated even without an environment success field.
        with patch('distillation.environments.Pool', MissingMetricPool):
            result = evaluate(cfg, make_adapters(cfg), teachers={'bounce': FakeTeacher()})
        self.assertEqual(result['bounce']['success_rate'], 1.)
        self.assertIsNone(result['bounce']['records'][0]['environment_success'])

    def test_zero_baseline_requires_explicit_option_and_can_train(self):
        cfg = self.config()
        teachers = {t: FakeTeacher() for t in cfg['tasks']}
        with tempfile.TemporaryDirectory() as folder, patch('distillation.environments.Pool', ZeroPool):
            output = Path(folder)
            with self.assertRaisesRegex(RuntimeError, 'allow-zero-teacher-success'):
                train(cfg, teachers, make_adapters(cfg), output)
            cfg['allow_zero_teacher_success'] = True
            train(cfg, teachers, make_adapters(cfg), output)
            self.assertTrue((output / 'student_last.pth').exists())
            report = json.loads((output / 'teacher_baseline.json').read_text())
            self.assertEqual(report['roll']['successes'], 0)


if __name__ == '__main__':
    unittest.main()
