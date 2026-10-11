import ast
import importlib.util
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('diagnostics',ROOT/'gym_dcmm/utils/failure_diagnostics.py')
D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)
class DiagnosticsTests(unittest.TestCase):
    def test_autoreset_terminal_record_and_failure_counts(self):
        agent=SimpleNamespace()
        def record(success):
            return json.dumps(dict(task='roll_catch',success=success,end='timeout',unmet=['hold_duration']))
        D.report(agent,dict(failure_diagnostic=[record(True)],final_info=[dict(failure_diagnostic=record(False))],_final_info=[True]),np.array([True]))
        D.report(agent,dict(failure_diagnostic=[record(True)]),np.array([True]))
        stats=agent._failure_diagnostic_totals['roll_catch']
        self.assertEqual(stats['episodes'],2)
        self.assertEqual(stats['failures'],1)
        self.assertEqual(stats['unmet']['hold_duration'],1)
    def test_finger_is_not_early_failure(self):
        tree=ast.parse((ROOT/'gym_dcmm/envs/DcmmVecEnv_bounce_july22.py').read_text(encoding='utf-8'))
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='bad_object_contact' for t in n.targets))
        for coll,finger,expected in [(False,True,False),(True,True,True),(False,False,False)]:
            scope=dict(np=np,self=SimpleNamespace(object_motion='bounce'),mask_coll=np.array([coll]),mask_finger=np.array([finger]))
            exec(compile(ast.Module(body=[node],type_ignores=[]),'<failure>','exec'),scope)
            self.assertEqual(bool(scope['bad_object_contact']),expected)
if __name__=='__main__': unittest.main()
