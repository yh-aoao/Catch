import unittest
from pathlib import Path
import numpy as np
import mujoco
from ycb_objects.benchmark_geometry import geometry
from ycb_objects.benchmark_randomization import sample,scaled_geometry
from gym_dcmm.envs.object_benchmark import make_object_benchmark,catalog

ROOT=Path(__file__).resolve().parents[1]
ALL=ROOT/'ycb_objects/calibration_data/object_benchmark_all_targets_v1.json'

class RandomizationTests(unittest.TestCase):
    def test_sampling_is_bounded_and_reproducible(self):
        for motion in ('roll','bounce'):
            a=sample(motion,np.random.RandomState(42));b=sample(motion,np.random.RandomState(42))
            self.assertEqual(a,b)
            for _ in range(100):
                d=sample(motion)
                for k,lo,hi in [('size_scale',.95,1.05),('mass_scale',.9,1.1),('friction_scale',.85,1.15)]:
                    self.assertTrue(lo<=d[k]<=hi)

    def test_mesh_scaling_and_inertia_without_mutating_cache(self):
        for name,kind in [('apple','convex'),('tomato_soup_can','cylinder'),('racquetball','sphere')]:
            base=geometry(name,kind);before=base['points'].copy()
            d=dict(size_scale=1.05,mass_scale=1.1)
            new=scaled_geometry(base,d)
            np.testing.assert_array_equal(base['points'],before)
            np.testing.assert_allclose(new['inertia'],base['inertia']*1.1*1.05**2)
            a=mujoco.MjModel.from_xml_string(base['xml']);b=mujoco.MjModel.from_xml_string(new['xml'])
            np.testing.assert_allclose(b.body_mass,a.body_mass*1.1,atol=1e-8)
            np.testing.assert_allclose(b.body_inertia,a.body_inertia*1.1*1.05**2,rtol=1e-5,atol=1e-9)
            if kind=='convex':
                # Compiled mesh vertices represent the actual convex collision hull.
                np.testing.assert_allclose(np.ptp(b.mesh_vert,axis=0),np.ptp(a.mesh_vert,axis=0)*1.05,rtol=1e-4)

    def test_all_targets_reset_and_step_with_fixed_sample(self):
        for motion in ('roll','bounce'):
            for name in catalog(ALL)[motion]:
                env=make_object_benchmark(object_benchmark=motion+'_v1',benchmark_catalog=ALL,
                    benchmark_options={'domain_randomization':'physical_dr_v1'},
                    object_motion=motion,task='Tracking',test_object=name,viewer=False,
                    render_mode=None,object_eval=False,benchmark_seed=123,benchmark_slot=0,benchmark_num_envs=1)
                try:
                    draw=dict(env.benchmark_randomization)
                    for seed in (10,11):
                        obs,info=env.reset(seed=seed)
                        self.assertEqual(info['object_benchmark']['randomization'],draw)
                        model=env.unwrapped.Dcmm.model
                        mass=float(model.body('object').mass[0])
                        self.assertTrue(mass>0)
                        action={k:np.zeros(v.shape,dtype=v.dtype) for k,v in env.action_space.spaces.items()}
                        _,r,_,_,_=env.step(action)
                        self.assertTrue(np.isfinite(r))
                        self.assertAlmostEqual(float(model.body('object').mass[0]),mass)
                finally:env.close()

if __name__=='__main__':unittest.main()
