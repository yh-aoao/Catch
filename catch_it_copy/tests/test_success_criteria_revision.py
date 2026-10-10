import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]

def step_nodes(path, cls):
    tree=ast.parse((ROOT/path).read_text(encoding='utf-8'))
    c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
    return next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='step').body

def run(node, env, info):
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual branch>','exec'),dict(self=env,info=info))

class SuccessTests(unittest.TestCase):
    def test_throw_official_stage_and_distance(self):
        nodes=step_nodes('gym_dcmm/envs/DcmmVecEnv.py','DcmmVecEnv')
        node=next(n for n in nodes if isinstance(n,ast.If) and ast.unparse(n.test)=="self.task == 'Catching' and self.object_motion == 'throw'")
        env=SimpleNamespace(task='Catching',object_motion='throw',stage='tracking',terminated=False)
        run(node,env,dict(ee_distance=.24))
        self.assertEqual(env.stage,'grasping')
        self.assertFalse(env.terminated)
        run(node,env,dict(ee_distance=.25))
        self.assertTrue(env.terminated)

    def test_throw_official_truncation_not_speed(self):
        nodes=step_nodes('gym_dcmm/envs/DcmmVecEnv.py','DcmmVecEnv')
        node=next(n for n in nodes if isinstance(n,ast.If) and "self.object_motion == 'throw_basket'"==ast.unparse(n.test) and 'truncated' in ast.unparse(n))
        code=compile(ast.Module(body=[node],type_ignores=[]),'<truncation>','exec')
        for task,touch,time,expected in [('Tracking',False,3.,False),('Tracking',True,1.,True),('Catching',False,2.4,False),('Catching',False,2.52,True)]:
            scope=dict(self=SimpleNamespace(task=task,object_motion='throw',step_touch=touch,env_time=2.5),info=dict(env_time=time))
            exec(code,scope)
            self.assertEqual(bool(scope['truncated']),expected)

    def test_bounce_track_timeout_and_failure(self):
        nodes=step_nodes('gym_dcmm/envs/DcmmVecEnv_bounce_july22.py','DcmmVecEnv')
        node=next(n for n in nodes if isinstance(n,ast.If) and 'bounce_touch_v1' in ast.unparse(n))
        for touch,failed,expected in [(False,False,False),(True,False,True),(True,True,False)]:
            info={}
            run(node,SimpleNamespace(task='Tracking',step_touch=touch,terminated=failed),info)
            self.assertEqual(info['success'],expected)

class BounceRevisionTests(unittest.TestCase):
    def test_catch_stage_and_timeout_match_throw(self):
        nodes=step_nodes('gym_dcmm/envs/DcmmVecEnv.py','BounceEnv')
        node=next(n for n in nodes if isinstance(n,ast.If) and ast.unparse(n.test)=="self.task == 'Catching' and self.object_motion == 'bounce'")
        env=SimpleNamespace(task='Catching',object_motion='bounce',stage='tracking',terminated=False)
        run(node,env,dict(ee_distance=.24))
        self.assertEqual(env.stage,'grasping')
        run(node,env,dict(ee_distance=.25))
        self.assertTrue(env.terminated)
        success_node=next(n for n in nodes if isinstance(n,ast.If) and 'bounce_throw_official' in ast.unparse(n))
        for timeout in (False,True):
            info={}
            exec(compile(ast.Module(body=[success_node],type_ignores=[]),'<success>','exec'),dict(self=env,info=info,truncated=timeout))
            self.assertEqual(info['success'],timeout)

    def test_tracking_finger_contact(self):
        import numpy as np
        tree=ast.parse((ROOT/'gym_dcmm/envs/DcmmVecEnv_bounce_july22.py').read_text(encoding='utf-8'))
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=='self.step_touch == False')
        for hand,palm,expected in [(True,False,True),(True,True,True),(False,False,False)]:
            env=SimpleNamespace(task='Tracking',object_motion='bounce',step_touch=False)
            scope=dict(self=env,np=np,mask_hand=np.array([hand]),mask_palm=np.array([palm]))
            exec(compile(ast.Module(body=[node],type_ignores=[]),'<contact>','exec'),scope)
            self.assertEqual(env.step_touch,expected)

if __name__=='__main__': unittest.main()
