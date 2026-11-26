import json,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from .eval_regions import circle_hit,summarize,make_env,draw_circles
class Checks(unittest.TestCase):
    def test_ball_clearance(self):
        self.assertTrue(circle_hit(.25,.6,.03));self.assertFalse(circle_hit(.28,.6,.03))
        self.assertFalse(circle_hit(None,1.,.03));self.assertFalse(circle_hit(0.,.01,.03))
    def test_denominator_safety_and_monotonicity(self):
        records=[dict(safe=True,crossing_radial=.35,release_stopped=False,original_success=False,valid_launch=True),
                 dict(safe=True,crossing_radial=.15,release_stopped=True,original_success=True,valid_launch=True),
                 dict(safe=False,crossing_radial=.1,release_stopped=True,original_success=False,valid_launch=True),
                 dict(safe=True,crossing_radial=None,release_stopped=False,original_success=False,valid_launch=False)]
        result=summarize(records,[.6,.8,1.],.03)
        self.assertEqual([r['hits'] for r in result['regions']],[1,2,2])
        self.assertEqual(result['regions'][1]['success_rate'],.5)
        self.assertEqual(result['regions'][1]['stopped_hits'],1)
        self.assertIsNone(result['smallest_tested_diameter_at_least_80pct'])
    def test_monitor_and_visual_geometry(self):
        import mujoco
        root=Path(__file__).resolve().parents[1]
        options=json.loads((root/'basket_mobile_bridge/configs/tennis.json').read_text())['options']
        options.pop('pose_bank');options.update(evaluation_task='far_mobile')
        env=make_env(options)
        try:
            env.reset(seed=123);size=env.Dcmm.model.geom_size.copy()
            scene=mujoco.MjvScene(env.Dcmm.model,maxgeom=200)
            draw_circles(SimpleNamespace(user_scn=scene),env,[.6,.8,1.])
            self.assertEqual(scene.ngeom,144);np.testing.assert_array_equal(size,env.Dcmm.model.geom_size)
            tilt=np.radians(env.basket_cfg.basket_tilt_deg);normal=np.array([0.,-np.sin(tilt),np.cos(tilt)])
            center=env.basket_center;cross=center+np.array([.2,0.,0.])
            env.Dcmm.data.qpos[37:40]=cross-.02*normal
            env._released_once=True;env._release=dict(confirmed=True,position=center-np.array([0.,.5,0.]),
                velocity=np.array([0.,2.,1.]),predicted_error=.1,valid_stop=True)
            with patch('basket_autonomous.environment.AutonomousTeacherEnv._basket_check_flight'),patch.object(env,'touching',return_value=False):
                env._basket_check_flight(cross+.02*normal)
            self.assertEqual(len(env.receiving_events),1);self.assertAlmostEqual(env.receiving_events[0]['radial'],.2)
        finally:env.close()
if __name__=='__main__':unittest.main()
