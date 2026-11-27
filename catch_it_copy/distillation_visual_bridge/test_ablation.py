"""Ablations remove modules and supervision, and preserve default checkpoints."""
import copy
import io
import unittest
import numpy as np
import torch
from .core import Student,ABLATIONS
from .run import optimize
from .test_sampling import AllObjectsTests

class AblationTests(unittest.TestCase):
    def setUp(self):
        fixture=AllObjectsTests();fixture.setUp();self.config=fixture.c
        self.batch=fixture.replay().sample(1,'cpu')
    def forward(self,model):
        b=self.batch
        return model(*[b[k] for k in ('depth','proprio','previous','context')],auxiliary=True)
    def test_default_is_full_and_parameter_identical(self):
        torch.manual_seed(123);default=Student(self.config)
        torch.manual_seed(123);full=Student(dict(self.config,ablation='full'))
        self.assertEqual(set(default.state_dict()),set(full.state_dict()))
        for key,value in default.state_dict().items():torch.testing.assert_close(value,full.state_dict()[key],rtol=0,atol=0)
        torch.testing.assert_close(self.forward(default)[0],self.forward(full)[0],rtol=0,atol=0)
    def test_removed_branches_ignore_their_labels_and_other_heads_train(self):
        for variant in ABLATIONS:
            with self.subTest(variant=variant):
                config=dict(self.config,ablation=variant);model=Student(config);batch=copy.deepcopy(self.batch)
                if not model.position_enabled:
                    batch['object_target'].fill_(float('nan'));batch['basket_target'].fill_(float('nan'))
                    self.assertIsNone(model.aux_head);self.assertIsNone(model.basket_position_head)
                if not model.future_enabled:
                    batch['future_target'].fill_(float('nan'))
                    self.assertIsNone(model.forecast);self.assertIsNone(model.fusion)
                metrics=optimize(model,torch.optim.Adam(model.parameters(),lr=1e-4),batch,config)
                self.assertTrue(all(np.isfinite(x) for x in metrics.values()))
                if not model.position_enabled:
                    self.assertEqual(metrics['position_loss'],0.);self.assertEqual(metrics['basket_position_loss'],0.)
                    self.assertFalse(any(k.startswith(('aux_head.','basket_position_head.')) for k in model.state_dict()))
                else:self.assertIsNotNone(model.aux_head[-1].weight.grad)
                if not model.future_enabled:
                    self.assertEqual(metrics['future_loss'],0.)
                    self.assertFalse(any(k.startswith(('forecast.','fusion.')) for k in model.state_dict()))
                else:
                    self.assertGreater(float(model.forecast[-1].weight.grad.abs().sum()),0.)
                    self.assertGreater(float(model.fusion.weight.grad.abs().sum()),0.)
                self.assertGreater(float(model.gru.weight_ih_l0.grad.abs().sum()),0.)
                self.assertGreater(float(model.basket_head[-1].weight.grad.abs().sum()),0.)
    def test_variant_save_load(self):
        for variant in ABLATIONS:
            config=dict(self.config,ablation=variant);model=Student(config).eval();buf=io.BytesIO()
            torch.save({'config':config,'student':model.state_dict()},buf);buf.seek(0)
            saved=torch.load(buf,weights_only=True);restored=Student(saved['config']).eval()
            restored.load_state_dict(saved['student'],strict=True)
            torch.testing.assert_close(self.forward(model)[0],self.forward(restored)[0],rtol=0,atol=0)

if __name__=='__main__':unittest.main()
