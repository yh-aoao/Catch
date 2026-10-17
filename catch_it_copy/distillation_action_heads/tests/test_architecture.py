import unittest
import torch
from distillation_action_heads.tests.test_pipeline import Tests as PipelineTests
from distillation_action_heads import run
class Tests(unittest.TestCase):
    def test_new_modules_receive_action_gradients(self):
        torch.set_num_threads(1)
        c=PipelineTests().configuration('catch');s=run.make_student(c)
        action=s(torch.rand(2,3,2,64,64),torch.randn(2,3,24),torch.randn(2,3,20))[0]
        (action-.3).square().mean().backward()
        self.assertGreater(sum(p.grad.abs().sum().item() for p in s.heads.parameters() if p.grad is not None),0)
        for encoder in s.encoders:
            self.assertGreater(encoder[0][0].weight.grad.abs().sum(),0)
