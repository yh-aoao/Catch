"""Rendering placement, resume compatibility, and collection timing contracts."""
import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from .runtime import validate_render_config,render_device,validate_resume
from .run import env_config,Collector
from .environments import Pool
from .core import SCHEMA

class PlacementTests(unittest.TestCase):
    def setUp(self):
        self.c=json.loads((Path(__file__).parent/'configs/catch.json').read_text())
    def test_round_robin_and_explicit_task_map(self):
        c=self.c;c['render_devices']=[1,2,3];validate_render_config(c)
        assigned=[render_device(c,t,i,8) for t in c['tasks'] for i in range(8)]
        self.assertEqual(assigned,[1,2,3]*10+[1,2])
        self.assertEqual(env_config(c,'roll')['render_devices'],[3,1,2,3,1,2,3,1])
        c.pop('render_devices');c['render_task_devices']=dict(throw=1,roll=1,bounce=2,basket=3)
        for task,device in c['render_task_devices'].items():
            self.assertEqual(env_config(c,task)['render_devices'],[device]*8)
    def test_invalid_and_legacy(self):
        self.assertIsNone(render_device(self.c,'throw',0,8))
        for values in ([],[-1],[1,1],['1']):
            c=copy.deepcopy(self.c);c['render_devices']=values
            with self.assertRaises(ValueError):validate_render_config(c)
        c=copy.deepcopy(self.c);c['render_task_devices']={'throw':1}
        with self.assertRaises(ValueError):validate_render_config(c)
        c=copy.deepcopy(self.c);c['render_devices']=[0];c['vision']['gl_backend']='glfw'
        with self.assertRaises(ValueError):validate_render_config(c)
    def test_resume_changes_only_hardware(self):
        metadata=dict(teacher_hashes={'throw':'same'},source_asset_sha256={})
        saved=dict(schema=SCHEMA,config=copy.deepcopy(self.c),metadata=metadata)
        current=copy.deepcopy(self.c);current['render_devices']=[1,2,3]
        self.assertEqual(validate_resume(saved,current,metadata,SCHEMA),( [],False))
        current['rollout_steps']+=1
        with self.assertRaises(ValueError):validate_resume(saved,current,metadata,SCHEMA)
    def test_pool_passes_device_to_each_spawned_worker(self):
        c=env_config(dict(self.c,render_devices=[1,2,3]),'throw');configs=[]
        class Pipe:
            def close(self):pass
        class Process:
            def start(self):pass
        class Context:
            def Pipe(self):return Pipe(),Pipe()
            def Process(self,**kwargs):
                configs.append(kwargs['args'][2]);return Process()
        with patch('multiprocessing.get_context',return_value=Context()),patch.object(Pool,'_receive',return_value={'x':np.zeros(1)}):
            pool=Pool('throw',c,8,123)
        self.assertEqual([cfg['render_device'] for cfg in configs],[1,2,3,1,2,3,1,2])
        self.assertEqual([cfg['selected_object'] for cfg in configs],['sphere','box','cylinder','capsule','ellipsoid','sphere','box','cylinder'])
    def test_collect_timings_do_not_change_actions_or_samples(self):
        from distillation.core import ActionAdapter
        c=self.c;collector=Collector.__new__(Collector)
        collector.config=c;collector.calibrating=False;collector.teacher_fraction=1.
        collector.student=type('Student',(),{'eval':lambda self:None})()
        packet={k:np.zeros((1,3),np.float32) for k in ('depth','proprio','object_target','object_valid','world_position','basket_target')}
        packet.update(context=np.zeros((1,10),np.float32),basket_valid=np.zeros(1,np.float32),basket_phase=np.zeros(1,np.float32),teacher_obs=np.zeros((1,30)),object_name=np.array(['sphere']))
        class FakePool:
            obs=packet
            def step_async(self,actions):self.actions=actions
            def step_wait(self):return packet,np.zeros(1),np.zeros(1,bool),[dict(module_hand_contact=False)]
        pool=FakePool();collector.pools={'throw':pool};collector.pending={'throw':[[]]};collector.previous={'throw':np.zeros((1,20),np.float32)}
        collector.hidden={'throw':torch.zeros(1,1,256)};collector.adapters={'throw':ActionAdapter(c['tasks']['throw'])}
        collector.teachers={'throw':type('Teacher',(),{'predict':lambda self,obs:np.full((1,20),.1,np.float32)})()}
        with patch('distillation_visual_bridge.run.student_step',return_value=(np.zeros((1,20),np.float32),collector.hidden['throw'])):
            timings=collector.collect(2)
        self.assertEqual(len(collector.pending['throw'][0]),2)
        np.testing.assert_allclose(collector.previous['throw'],.1)
        self.assertTrue(all(v>=0 for v in timings.values()))
        self.assertAlmostEqual(timings['collect_seconds'],sum(timings[k] for k in ('student_inference_seconds','teacher_inference_seconds','environment_wait_seconds','collect_other_seconds')))

if __name__=='__main__':unittest.main()
