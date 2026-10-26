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
    def test_catch_requires_grasp_stage_near_ball_and_no_failure(self):
        nodes=step_nodes('gym_dcmm/envs/DcmmVecEnv.py','BounceEnv')
        node=next(n for n in nodes if isinstance(n,ast.If) and ast.unparse(n.test)=="self.task == 'Catching' and self.object_motion == 'bounce'")
        env=SimpleNamespace(task='Catching',object_motion='bounce',stage='tracking',terminated=False)
        run(node,env,dict(ee_distance=.24))
        self.assertEqual(env.stage,'grasping')
        run(node,env,dict(ee_distance=.25))
        self.assertTrue(env.terminated)
        success_node=next(n for n in nodes if isinstance(n,ast.If) and 'bounce_catch_proximity_v2' in ast.unparse(n))
        import numpy as np
        cases = [('tracking',1.0417,False,True,False),
                 ('tracking',1.2164,False,True,False),
                 ('tracking',.1,False,True,False),
                 ('grasping',.24,False,True,True),
                 ('grasping',.25,False,True,False),
                 ('grasping',.1,True,True,False),
                 ('grasping',.1,False,False,False),
                 ('grasping',float('nan'),False,True,False)]
        for stage,distance,failed,timeout,expected in cases:
            with self.subTest(stage=stage,distance=distance,failed=failed,timeout=timeout):
                env=SimpleNamespace(task='Catching',object_motion='bounce',stage=stage,
                                    terminated=failed,terminated_reason='ball_left' if failed else None)
                info={'ee_distance':distance}
                exec(compile(ast.Module(body=[success_node],type_ignores=[]),'<success>','exec'),
                     dict(self=env,info=info,truncated=timeout,np=np))
                self.assertEqual(info['success'],expected)
                self.assertEqual(info['bounce_legacy_timeout_success'],timeout)

    def test_actual_dispatch_separates_catch_from_track(self):
        import copy
        import sys
        from types import ModuleType
        from unittest.mock import patch
        tree=ast.parse((ROOT/'gym_dcmm/envs/DcmmVecEnv.py').read_text(encoding='utf-8'))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DcmmVecEnv')
        route=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__new__'))
        stub=ast.parse("class DcmmVecEnv:\n def __init__(self, task='Tracking', object_motion='throw'): pass\n").body[0]
        stub.body.append(route)
        factory=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_make_main_bounce')
        scope={'BounceEnv':lambda **kw: SimpleNamespace(route='5fe75d5f',parameters=kw)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[factory,stub],type_ignores=[])),'<route>','exec'),scope)
        compat=ModuleType('gym_dcmm.envs.bounce_compat')
        compat.make_bounce_july22=lambda kw: SimpleNamespace(route='july22')
        with patch.dict(sys.modules,{compat.__name__:compat}):
            for alias in ('bounce','tan','弹'):
                env=scope['DcmmVecEnv'](task='Catching',object_motion=alias)
                self.assertEqual(env.route,'5fe75d5f')
                self.assertEqual(env.parameters['object_motion'],'bounce')
                self.assertEqual(scope['DcmmVecEnv'](task='Tracking',object_motion=alias).route,'july22')

    def test_original_ppo_uses_false_success_in_terminal_info(self):
        import numpy as np
        tree=ast.parse((ROOT/'gym_dcmm/algs/ppo_dcmm/ppo_dcmm_track.py').read_text(encoding='utf-8'))
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='terminal_metrics')
        scope={'np':np}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<ppo metrics>','exec'),scope)
        success,reasons=scope['terminal_metrics'](
            {'final_info':[{'success':False,'terminated_reason':'timeout_not_caught'}]},
            np.array([True]),np.array([True]))
        self.assertFalse(success[0])
        self.assertEqual(reasons[0],'timeout_not_caught')

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
