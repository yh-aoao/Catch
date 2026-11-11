"""Regression checks for actual per-episode changes, not just sampled metadata."""
import unittest
from pathlib import Path
import numpy as np
from gym_dcmm.envs.object_benchmark import make_object_benchmark
ROOT=Path(__file__).resolve().parents[1]

class EpisodeRandomizationTests(unittest.TestCase):
    def make(self,motion,name,version='physical_dr_episode_v1'):
        return make_object_benchmark(object_benchmark=motion+'_v1',
            benchmark_catalog=ROOT/'ycb_objects/calibration_data/object_benchmark_all_targets_v1.json',
            benchmark_options={'domain_randomization':version},object_motion=motion,
            task='Catching',test_object=name,viewer=False,render_mode=None,object_eval=False,
            benchmark_slot=0,benchmark_num_envs=1,benchmark_seed=123,
            bounce_grasp_version='controlled_hold_v1')

    def test_resample_reproduce_and_scale_without_drift(self):
        for motion,name in [('roll','apple'),('roll','tomato_soup_can'),('bounce','original')]:
            with self.subTest(motion=motion,name=name):
                env=self.make(motion,name)
                try:
                    records=[]
                    for seed in (891,892,891):
                        _,info=env.reset(seed=seed);d=env.benchmark_randomization.copy();m=env.unwrapped.Dcmm.model
                        np.testing.assert_allclose(m.body_mass[env.bid],env.mass*d['mass_scale'])
                        np.testing.assert_allclose(m.body_inertia[env.bid],env.inertia*d['mass_scale']*d['size_scale']**2)
                        np.testing.assert_allclose(m.geom_aabb[env.ids],env.nominal['geom_aabb']*d['size_scale'])
                        self.assertEqual(info['object_benchmark']['randomization'],d)
                        if name=='apple':
                            leaf=env.unwrapped;low,high=leaf._benchmark_world_bounds()
                            from ycb_objects.benchmark_geometry import bounds
                            a,b=bounds(leaf.benchmark_shape,leaf.Dcmm.data.body('object').xmat.reshape(3,3))
                            np.testing.assert_allclose(low,leaf.Dcmm.data.body('object').xpos+a)
                            np.testing.assert_allclose(high,leaf.Dcmm.data.body('object').xpos+b)
                        records.append((d,m.geom_size[env.ids].copy()))
                    self.assertNotEqual(records[0][0],records[1][0]);self.assertEqual(records[0][0],records[2][0])
                    np.testing.assert_array_equal(records[0][1],records[2][1])
                finally:env.close()

    def test_old_dr_keeps_creation_sample(self):
        env=self.make('roll','apple','physical_dr_v1')
        try:
            draw=env.benchmark_randomization.copy()
            for seed in (891,892):
                _,info=env.reset(seed=seed)
                self.assertEqual(draw,env.benchmark_randomization)
                self.assertEqual(draw,info['object_benchmark']['randomization'])
        finally:env.close()

if __name__=='__main__':unittest.main()
