import copy,json,unittest
from pathlib import Path
import numpy as np
import torch
from .environment import AutonomousTeacherEnv
from .policy import Policy
from .run import validate_initialization,package
from gym_dcmm.utils.basket_catching import hoop_crossing
ROOT=Path(__file__).resolve().parents[1]
class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config=json.loads((ROOT/'basket_mobile_bridge_large/configs/tennis_d060.json').read_text())
        o=cls.config['options'];o['pose_bank_entries']=json.loads((ROOT/o.pop('pose_bank')).read_text())
    def test_rim_scoring_reset_and_global_isolation(self):
        from configs.env import DcmmCfg
        original=DcmmCfg.basket_radius
        env=AutonomousTeacherEnv(dict(self.config['options'],evaluation_task='mobile'))
        try:
            for seed in [123,124]:
                obs,_=env.reset(seed=seed);self.assertEqual(obs.shape,(66,));self.assertEqual(obs[-1],1.)
                self.assertEqual(env.basket_cfg.basket_radius,.3)
                m=env.Dcmm.model
                ids=[m.geom('basket_seg_'+str(i)).id for i in range(24)]
                np.testing.assert_allclose(np.linalg.norm(m.geom_pos[ids,:2],axis=1),.3,atol=1e-10)
                np.testing.assert_allclose(m.geom_size[ids,0],.01)
                self.assertTrue(np.all(m.geom_contype[ids]==1))
                self.assertEqual(DcmmCfg.basket_radius,original)
            tilt=np.radians(env.basket_cfg.basket_tilt_deg);normal=np.array([0.,-np.sin(tilt),np.cos(tilt)])
            center=env.basket_center;point=center+np.array([.2,0.,0.])
            self.assertTrue(hoop_crossing(point+.1*normal,point-.1*normal,center,env.basket_cfg.basket_tilt_deg,.3,env.radius)[1])
            self.assertFalse(hoop_crossing(point+.1*normal,point-.1*normal,center,env.basket_cfg.basket_tilt_deg,original,env.radius)[1])
        finally:env.close()
    def test_existing_bridge_import(self):
        saved=torch.load(ROOT/'basket_mobile_bridge/runs/tennis_bridge_seed123/teacher_last.pth',map_location='cpu',weights_only=False)
        validate_initialization(saved,self.config);model=Policy();model.load_state_dict(saved['model'],strict=True)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertEqual(package(model,self.config,0)['schema'],'basket_mobile_bridge_large_v1')
    def test_switch_criteria_unchanged(self):
        from basket_mobile_bridge.environment import ready_to_throw
        info=dict(hand_contact=True,base_speed=.01,carry_diagnostics=dict(slip_speed=.01,in_support_envelope=True))
        self.assertTrue(ready_to_throw(info,1.,0.));self.assertFalse(ready_to_throw(info,1.5,0.))
        env=AutonomousTeacherEnv(dict(self.config['options'],evaluation_task='mobile',episode_seconds=.04))
        try:
            env.reset(seed=123);obs,r,done,truncated,info=env.step(np.zeros(20))
            self.assertTrue(done or truncated);self.assertTrue(np.isfinite(r));self.assertEqual(obs.shape,(66,))
        finally:env.close()
if __name__=='__main__':unittest.main()
