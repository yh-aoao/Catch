import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from .run import evaluate
from distillation.core import ActionAdapter

class HitMetricTests(unittest.TestCase):
    def test_same_trajectory_hit_success_and_strict_failure(self):
        config=json.loads((Path(__file__).parent/'configs/catch.json').read_text())
        config['eval_episodes']=1
        info=dict(basket_region_hit=True,basket_stopped_region_hit=False,safe_basket_success=False,
                  released=True,release_stopped=False,reached_throw_phase=True,terminated_reason='ball_on_floor')
        class Pool:
            obs={}
            def __init__(self,*args,**kwargs):pass
            def step(self,actions):return {},np.array([0.]),np.array([True]),[copy.deepcopy(info)]
            def close(self):pass
        class Student:
            def eval(self):pass
        adapters={'basket':ActionAdapter(config['tasks']['basket'])}
        with tempfile.TemporaryDirectory() as directory:
            for metric,successes in [('basket_region_hit',1),('safe_basket_success',0)]:
                config['tasks']['basket']['success_metric']=metric
                with patch('distillation_visual_bridge.run.Pool',Pool),patch('distillation_visual_bridge.run.student_step',return_value=(np.zeros((1,20),np.float32),torch.zeros(1,1,256))):
                    result=evaluate(config,adapters,Student(),Path(directory)/'result.json',tasks=['basket'])
                self.assertEqual(result['basket']['successes'],successes)
                self.assertEqual(result['basket']['success_metric'],metric)
                diag=result['basket']['objects']['tennis_ball']['diagnostics']
                self.assertEqual(diag['basket_region_hit'],1)
                self.assertEqual(diag['safe_basket_success'],0)

if __name__=='__main__':unittest.main()
