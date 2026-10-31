"""CPU synthetic checks, including training/checkpoint/resume without MuJoCo."""
import unittest
from unittest.mock import patch
import torch
from visual_module_experiments import test_modules as existing
from visual_module_experiments.scaffold import VARIANTS
from . import run


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_schedule(self):
        config = dict(eval_start=5000, eval_every=1000, iterations=10000)
        self.assertEqual([i for i in range(1,10001) if run.should_evaluate(i,config)],
                         [5000,6000,7000,8000,9000,10000])

    def test_fusion_sequence_and_gradients(self):
        config = existing.Tests().configuration('future_contact')
        student = run.make_student(config)
        x = (torch.rand(1,6,2,64,64), torch.rand(1,6,24), torch.rand(1,6,20))
        action, _, _ = student(*x)
        hidden = None
        pieces = []
        for j in range(6):
            a, hidden, _ = student(*(v[:,j:j+1] for v in x),hidden)
            pieces.append(a)
        torch.testing.assert_close(action,torch.cat(pieces,1),atol=1e-6,rtol=1e-5)
        batch = dict(depth=x[0],proprio=x[1],previous=x[2],target=torch.zeros_like(action),
                     valid=torch.ones(1,6),object_target=torch.ones(1,6,6),object_valid=torch.ones(1,6,6),
                     future_target=torch.ones(1,6,3,3),future_valid=torch.ones(1,6,3),
                     contact_target=torch.ones(1,6,3),contact_valid=torch.ones(1,6,3))
        optimizer = torch.optim.Adam(student.parameters(),lr=.001)
        # Zero-initialized Contact modulation acquires gradients on the first step.
        for _ in range(2):
            losses = run.optimize(student,optimizer,batch,config)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in losses.values()))
        for name in ('forecast','fusion','phase','modulation','gate'):
            self.assertGreater(sum(p.grad.abs().sum().item() for p in getattr(student,name).parameters()),0)

    def test_existing_mask_and_train_resume_contracts(self):
        helper = existing.Tests()
        original = helper.configuration
        def configuration(variant, stage='catch'):
            config = original(variant,stage)
            config['eval_start'] = 1
            config['skip_teacher_baseline'] = True
            return config
        helper.configuration = configuration
        with patch.dict(VARIANTS, {'future_contact':'distillation_aux_future_contact'},clear=True):
            helper.test_extra_loss_padding_mask()
            helper.test_train_save_eval_resume_all_variants_stages()


if __name__ == '__main__':
    unittest.main()
