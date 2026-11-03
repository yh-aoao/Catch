import unittest
import random
from collections import Counter
from pathlib import Path
import numpy as np
from gym_dcmm.envs.object_benchmark import catalog,choose,make_benchmark_vector,make_object_benchmark

ALL=Path(__file__).resolve().parents[1]/'ycb_objects/calibration_data/object_benchmark_all_targets_v1.json'

class AllTargetsTests(unittest.TestCase):
    def test_catalog_and_balanced_slots(self):
        d=catalog(ALL)
        expected={'roll':{'original','tennis_ball','baseball','orange','apple','lemon','tomato_soup_can'},'bounce':{'original','tennis_ball','baseball','racquetball','golf_ball','softball'}}
        for motion,n in [('roll',28),('bounce',24)]:
            self.assertEqual(set(d[motion]),expected[motion])
            counts=Counter(choose(motion,'default','mixed',None,d,i,n) for i in range(n))
            self.assertEqual(set(counts.values()),{4})
        self.assertEqual(len(catalog()['roll']),4)
        self.assertEqual(len(catalog()['bounce']),3)

    def test_async_all_object_slots_and_step(self):
        for motion,n in [('roll',7),('bounce',6)]:
            env=make_benchmark_vector(n,benchmark_backend='async',object_benchmark=motion+'_v1',benchmark_catalog=str(ALL),object_motion=motion,task='Tracking',test_object='default',viewer=False,render_mode=None,object_eval=False,**{motion+'_ball_mode':'mixed'})
            try:
                self.assertEqual(set(env.call('benchmark_object')),set(catalog(ALL)[motion]))
                obs,info=env.reset(seed=123)
                actions={k:np.zeros((n,)+s.shape,dtype=s.dtype) for k,s in env.single_action_space.spaces.items()}
                obs,rewards,terminated,truncated,info=env.step(actions)
                self.assertTrue(np.isfinite(rewards).all())
            finally:env.close()

    def test_legacy_three_ball_routes_preserve_reset_and_dynamics(self):
        for motion in ('roll','bounce'):
            for name in ('original','tennis_ball','baseball'):
                test_object='default' if name=='original' else 'ycb:'+name
                common=dict(task='Tracking',object_motion=motion,viewer=False,render_mode=None,object_eval=False,env_time=2.5,steps_per_policy=20)
                combined=make_object_benchmark(object_benchmark=motion+'_v1',benchmark_catalog=str(ALL),test_object=test_object,**common)
                if motion=='roll':
                    from gym_dcmm.envs.roll_physical import make_roll_physical,DEFAULT_MATERIALS
                    original=make_roll_physical(test_object=test_object,roll_speed=[.8,1.3],roll_material_file=str(DEFAULT_MATERIALS),**common)
                else:
                    from gym_dcmm.envs.bounce_physical_v2 import make_bounce_physical_v2
                    original=make_bounce_physical_v2(test_object=test_object,bounce_launch='baseball_intercept_v1',bounce_material_file=str(ALL.parent/'balls.three_ball_training_v1.json'),**common)
                try:
                    combined.reset(seed=77);original.reset(seed=77)
                    np.testing.assert_array_equal(combined.unwrapped.Dcmm.data.qpos,original.Dcmm.data.qpos)
                    np.testing.assert_array_equal(combined.unwrapped.Dcmm.data.qvel,original.Dcmm.data.qvel)
                    actions={k:np.zeros(s.shape,dtype=s.dtype) for k,s in original.action_space.spaces.items()}
                    for step in range(3):
                        # These environments deliberately use global RNGs for
                        # observation/action noise. Pair the disturbances too.
                        np.random.seed(1000+step);random.seed(1000+step)
                        a=combined.step(actions)
                        np.random.seed(1000+step);random.seed(1000+step)
                        b=original.step(actions)
                        self.assertEqual(a[1:4],b[1:4])
                        np.testing.assert_array_equal(combined.unwrapped.Dcmm.data.qpos,original.Dcmm.data.qpos)
                finally:combined.close();original.close()

if __name__=='__main__':unittest.main()
