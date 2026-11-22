import json
from pathlib import Path
import unittest
import numpy as np
import torch
from .core import Student,action_mask,command_context,Replay,losses

class StudentContract(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.c=json.loads((Path(__file__).parent/'configs/catch.json').read_text())
        self.s=Student(self.c)
    def test_no_geometry_or_stage_can_enter_forward(self):
        depth=torch.zeros(1,2,*self.s.image_shape,dtype=torch.uint8)
        state=torch.zeros(1,2,36);previous=torch.zeros(1,2,20);context=torch.zeros(1,2,10);context[...,0]=1
        action,_,_=self.s(depth,state,previous,context)
        self.assertEqual(action.shape,(1,2,20))
        self.assertTrue(torch.all(action_mask(context)==1))
        self.assertEqual(self.s.extra['basket_position'].shape,(1,2,3))
        self.s.extra['basket_position'].square().sum().backward()
        self.assertIsNotNone(self.s.gru.weight_ih_l0.grad)
        context[...,1]=.1
        with self.assertRaises(ValueError):self.s(depth,state,previous,context)
    def test_command_is_only_mode(self):
        obs={'base':{'v_lin_2d':np.zeros((3,2))}}
        context=command_context(obs,'basket')
        self.assertTrue(np.all(context[:,0]==1));self.assertTrue(np.all(context[:,1:]==0))
    def test_fixed_batch_size_and_equal_task_weight(self):
        r=Replay(self.c)
        for t,o in r.groups:
            e={k:torch.zeros(2,1) for k in r.keys}
            e.update(depth=torch.zeros(2,2,160,160,dtype=torch.uint8),world_position=torch.zeros(2,3),contact_after=torch.zeros(2))
            e['context']=torch.zeros(2,10);e['context'][:,0]=float(t=='basket')
            r.add_episode(t+'/'+o,e)
        b=r.sample(1,'cpu')
        self.assertEqual(len(b['weights']),8)
        for i in range(4):self.assertAlmostEqual(float(b['weights'][b['task_id']==i].sum()),.25)

if __name__=='__main__':unittest.main()
