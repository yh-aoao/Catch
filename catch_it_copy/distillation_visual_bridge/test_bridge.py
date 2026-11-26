"""Physical geometry and direct control contracts."""
import json
import unittest
from pathlib import Path
import numpy as np
from distillation.core import ActionAdapter
from .factories import make_basket
from .teachers import FrozenTeacher
from .run import validate

class BridgeContracts(unittest.TestCase):
    def test_teacher_and_control(self):
        c=json.loads((Path(__file__).parent/'configs/catch.json').read_text());validate(c)
        cfg=c['tasks']['basket'];teacher=FrozenTeacher('basket',cfg,'cpu')
        self.assertEqual(teacher.parity()['inference_max_error'],0.)
        env=make_basket(**cfg['env_kwargs'])
        try:
            obs,_=env.reset(seed=123)
            self.assertEqual(obs.shape,(66,))
            self.assertAlmostEqual(env.basket_cfg.basket_radius,.3)
            for i in range(24):
                geom=env.Dcmm.model.geom('basket_seg_'+str(i))
                self.assertAlmostEqual(float(np.linalg.norm(geom.pos[:2])),.3,delta=1e-4)
            self.assertTrue(env.basket_integrated_arm_velocity)
            action=np.zeros(20,np.float32);action[2:8]=.01
            env.step(ActionAdapter(cfg).action_dict(action))
            np.testing.assert_allclose(env.last_action,action,atol=1e-7)
            self.assertFalse(env.fixed_episode)
        finally:env.close()

if __name__=='__main__':unittest.main()
