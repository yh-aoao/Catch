import json,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
from .environment import AutonomousTeacherEnv,preparation_metrics
ROOT=Path(__file__).resolve().parents[1]
class MovingTests(unittest.TestCase):
    def test_ready_does_not_require_stopping(self):
        model=NS(jnt_qposadr=np.arange(15,21),jnt_range=np.tile([-2.,2.],(6,1)),jnt_limited=np.ones(6,bool))
        data=NS(qpos=np.zeros(44),qvel=np.zeros(42),body=lambda name:NS(xmat=np.eye(3).ravel(),xpos=np.zeros(3)))
        env=NS(Dcmm=NS(data=data,model=model),basket_center=np.array([0.,1.,.55]),_released_once=False,
            _navigation_distance=lambda:0.,_low_speed_time=0.)
        info=dict(hand_contact=True,base_speed=.3,carry_diagnostics=dict(slip_speed=.01))
        with patch('basket_prepare_throw_d060.environment.palm_coordinates',return_value=np.array([0.,.03,.08])):
            self.assertTrue(preparation_metrics(env,info)['ready'])
    def test_physical_hoop_and_moving_launch_score(self):
        options=json.loads((ROOT/'basket_prepare_throw_d060/configs/stage1_prepare.json').read_text())['options']
        options.pop('pose_bank');options['evaluation_task']='prepare'
        env=AutonomousTeacherEnv(options)
        try:
            env.reset(seed=123);m,d=env.Dcmm.model,env.Dcmm.data
            self.assertEqual(env.basket_cfg.basket_radius,.3)
            ids=[m.geom('basket_seg_'+str(i)).id for i in range(24)]
            np.testing.assert_allclose(np.linalg.norm(m.geom_pos[ids,:2],axis=1),.3,atol=1e-10)
            center=env.basket_center;release=dict(confirmed=True,position=center-np.array([0.,.5,0.]),velocity=np.array([0.,2.,1.]),
                time=float(d.time)-.2,valid_stop=False,distance=.5,predicted_error=.1)
            env._release=release;env._last_touch=release.copy();env._released_once=True;env._no_hand_seconds=.2
            tilt=np.radians(env.basket_cfg.basket_tilt_deg);normal=np.array([0.,-np.sin(tilt),np.cos(tilt)])
            point=center+np.array([.1,0.,0.]);d.qpos[37:40]=point-.01*normal
            with patch.object(env,'_get_contacts',return_value=dict(object_contacts=np.array([],int))):
                env._basket_check_flight(point+.01*normal)
            self.assertTrue(env._scored);self.assertFalse(env._release['valid_stop'])
            # Non-stopped confirmed intentional release must also receive launch credit.
            env._scored=False;env._released_once=False;env._release=dict(release,confirmed=False)
            with patch.object(env,'_hand_clearance',return_value=1.):
                env._update_confirmed_release(False,point,0.3,.02,{1})
            self.assertTrue(env._released_once);self.assertGreater(env._release_reward,0.)
        finally:env.close()
if __name__=='__main__':unittest.main()
