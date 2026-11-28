import copy
import unittest
import torch
from .core import Student,losses
from .runtime import load_initial_weights
from .test_sampling import AllObjectsTests

class ThrowRepairTests(unittest.TestCase):
    def setUp(self):
        fixture=AllObjectsTests();fixture.setUp()
        self.config=copy.deepcopy(fixture.c)
        self.batch=fixture.replay().sample(1,'cpu')

    def test_new_auxiliary_preserves_actions_on_transfer_and_gets_gradients(self):
        original=Student(self.config)
        config=dict(self.config,basket_phase_weight=.1,basket_throw_precision_weight=1.)
        model=Student(config)
        missing=load_initial_weights(model,original.state_dict())
        self.assertTrue(missing);self.assertTrue(all(k.startswith('basket_phase_head.') for k in missing))
        x=[self.batch[k] for k in ('depth','proprio','previous','context')]
        expected=original(*x)[0]
        action,_,logits,prediction=model(*x,auxiliary=True)
        torch.testing.assert_close(action,expected,rtol=0,atol=0)
        loss=losses(model,self.batch,0,2,logits,prediction,config,self.batch['valid'].sum(1))
        self.assertGreater(float(loss['basket_phase_loss'].detach()),0)
        self.assertGreater(float(loss['basket_throw_precision_loss'].detach()),0)
        loss['loss'].backward()
        self.assertGreater(float(model.basket_phase_head[-1].weight.grad.abs().sum()),0)
        self.assertTrue(torch.isfinite(model.basket_head[-1].weight.grad).all())
        bad=dict(original.state_dict());bad.pop('basket_head.0.weight')
        with self.assertRaises(ValueError):load_initial_weights(model,bad)

    def test_precision_masks_carry_and_catch_and_phase_masks_catch(self):
        config=dict(self.config,basket_phase_weight=.1,basket_throw_precision_weight=1.)
        model=Student(config);batch=self.batch
        x=[batch[k] for k in ('depth','proprio','previous','context')]
        _,_,logits,prediction=model(*x,auxiliary=True)
        lengths=batch['valid'].sum(1)
        batch['basket_phase'].zero_()
        loss=losses(model,batch,0,2,logits,prediction,config,lengths)
        self.assertEqual(float(loss['basket_throw_precision_loss'].detach()),0.)
        batch['context'].zero_();batch['basket_phase'].fill_(1.)
        loss=losses(model,batch,0,2,logits,prediction,config,lengths)
        self.assertEqual(float(loss['basket_throw_precision_loss'].detach()),0.)
        self.assertEqual(float(loss['basket_phase_loss'].detach()),0.)

if __name__=='__main__':unittest.main()
