import unittest
import numpy as np
import torch
from distillation_feature.features import predict_with_features, feature_loss
from distillation_feature.tests.test_pipeline import Teacher, Tests as PipelineTests
from distillation_feature import run


class Tests(unittest.TestCase):
    def test_feature_hooks_do_not_change_actions_or_remain_installed(self):
        for stage in ('track', 'catch'):
            teacher = Teacher(stage)
            raw = np.ones((2, 18 if stage == 'track' else 30), np.float32)
            expected = teacher.predict(raw)
            action, feature = predict_with_features(teacher, raw, 512)
            np.testing.assert_array_equal(expected, action)
            np.testing.assert_allclose(np.linalg.norm(feature, axis=1), 1., atol=1e-6)
            self.assertTrue(all(not m._forward_hooks for m in teacher.model.modules()))
            with self.assertRaises(ValueError):
                predict_with_features(teacher, raw, 1)
            self.assertTrue(all(not m._forward_hooks for m in teacher.model.modules()))

    def test_feature_heads_train_but_do_not_change_deployment_actions(self):
        torch.set_num_threads(1)
        c = PipelineTests().configuration('catch')
        s = run.make_student(c)
        inputs = (torch.rand(1, 2, 2, 64, 64), torch.zeros(1, 2, 24), torch.zeros(1, 2, 20))
        before = s(*inputs)[0].detach()
        loss = feature_loss(s, torch.randn(1, 2, 512), torch.zeros(1, 2, dtype=torch.long), torch.ones(1, 2), torch.tensor([2.]))
        loss.backward()
        self.assertGreater(s.gru.weight_ih_l0.grad.abs().sum(), 0)
        with torch.no_grad():
            for p in s.feature_heads.parameters():
                p.add_(10)
        torch.testing.assert_close(before, s(*inputs)[0])


if __name__ == '__main__':
    unittest.main()
