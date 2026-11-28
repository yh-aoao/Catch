import copy
import unittest
import numpy as np
import torch
from .runtime import latch_episode_choices,validate_initial_student
from .core import Student,losses
from .test_sampling import AllObjectsTests

class CarryRepairTests(unittest.TestCase):
    def test_executor_latches_until_reset(self):
        latched,choice=latch_episode_choices(np.array([True,False]))
        np.testing.assert_array_equal(choice,[True,False])
        latched,choice=latch_episode_choices(np.array([False,True]),latched)
        np.testing.assert_array_equal(choice,[True,False])
        latched[0]=-1
        latched,choice=latch_episode_choices(np.array([False,True]),latched)
        np.testing.assert_array_equal(choice,[False,False])
    def test_weight_initialization_contract(self):
        fixture=AllObjectsTests();fixture.setUp();config=fixture.c
        hashes={task:item['sha256'] for task,item in config['tasks'].items()}
        saved=dict(schema='schema',config=copy.deepcopy(config),metadata={'teacher_hashes':hashes})
        config['basket_rollin']='episode';config['basket_carry_precision_weight']=10.
        config['tasks']['basket']['success_metric']='basket_region_hit'
        validate_initial_student(saved,config,'schema',hashes)
        config['ablation']='none'
        with self.assertRaises(ValueError):validate_initial_student(saved,config,'schema',hashes)

    def test_precision_only_supervises_basket_carry(self):
        fixture=AllObjectsTests();fixture.setUp();config=fixture.c;batch=fixture.replay().sample(1,'cpu')
        model=Student(config);x=[batch[k] for k in ('depth','proprio','previous','context')]
        _,_,logits,prediction=model(*x,auxiliary=True)
        lengths=batch['valid'].sum(1)
        original=losses(model,batch,0,2,logits,prediction,config,lengths)
        repair=losses(model,batch,0,2,logits,prediction,dict(config,basket_carry_precision_weight=10.),lengths)
        torch.testing.assert_close(repair['loss'],original['loss']+10*repair['basket_carry_precision_loss'])
        self.assertGreater(float(repair['basket_carry_precision_loss'].detach()),0)
        repair['loss'].backward();self.assertTrue(torch.isfinite(model.basket_head[-1].weight.grad).all())
        batch['basket_phase'].fill_(1.)
        _,_,logits,prediction=model(*x,auxiliary=True)
        throwing=losses(model,batch,0,2,logits,prediction,dict(config,basket_carry_precision_weight=10.),lengths)
        self.assertEqual(float(throwing['basket_carry_precision_loss'].detach()),0.)
        self.assertNotIn('basket_carry_precision_loss',original)

if __name__=='__main__':unittest.main()
