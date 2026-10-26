import unittest
from types import SimpleNamespace as N
from unittest.mock import patch
import numpy as np
from gym_dcmm.utils.basket_diagnostics import ArmDiagnostics


class DiagnosticsTests(unittest.TestCase):
    def test_contact_probe_and_read_only(self):
        m = N(nbody=4, body_parentid=np.array([0,0,1,2]),
              geom_bodyid=np.array([1,2,3]), geom_contype=np.array([1,1,1]),
              geom_conaffinity=np.array([1,1,1]),
              body=lambda i: N(name=['world','link1','link2','link3'][i]),
              geom=lambda i: N(name='g'+str(i)))
        d = N(contact=[N(geom1=0,geom2=2,dist=-.001)], qpos=np.arange(44,dtype=float))
        original = d.qpos.copy()
        fake = N(mj_geomDistance=lambda *args: -.001,
                 mj_contactForce=lambda m,d,i,f: f.__setitem__(0, 2.))
        with patch('builtins.print') as output, patch.dict('sys.modules', {'mujoco':fake}):
            diag = ArmDiagnostics(m)
            self.assertEqual(diag.pairs, [(0,2,False)])
            diag.sample(d, .1)
            text = str(output.call_args_list)
            self.assertIn('normal_force', text)
            self.assertIn('geometry_proximity', text)
            count = output.call_count
            diag.sample(d, .15)
            self.assertEqual(output.call_count, count)
        np.testing.assert_array_equal(original, d.qpos)
