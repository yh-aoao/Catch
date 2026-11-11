"""Geometry, physical motion initialization and fork-safe scene allocation."""
import unittest
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from ycb_objects.benchmark_geometry import geometry,roll_state
from gym_dcmm.envs.object_benchmark import catalog,choose,bounce_launch,make_benchmark_vector,make_object_benchmark

class ObjectBenchmarkTests(unittest.TestCase):
    def test_fruit_collision_and_inertia(self):
        for name in ('orange','apple','lemon'):
            s=geometry(name,'convex');body=ET.fromstring(s['xml']).find('./worldbody/body')
            self.assertEqual(body.find("geom[@name='benchmark_collision']").get('type'),'mesh')
            self.assertEqual(body.find("geom[@name='object']").get('contype'),'0')
            self.assertGreater(len(s['points']),100)
            self.assertTrue(np.all((s['inertia']>0)&(s['inertia']<.001)))

    def test_can_sideways_and_no_slip(self):
        s=geometry('tomato_soup_can','cylinder')
        for heading in (-.15,0,.15):
            a=roll_state(s,1.,heading,.7);axis=a['rotation'][:,2]
            self.assertAlmostEqual(axis[2],0.,places=10)
            self.assertAlmostEqual(float(axis@a['velocity'][:3]),0.,places=10)
            self.assertAlmostEqual(a['clearance'],s['radius'],places=10)
            self.assertLess(a['no_slip_error'],1e-10)
        self.assertTrue(np.isfinite(roll_state(s,0,0,.7)['velocity']).all())

    def test_no_slip_in_mujoco_world_velocity(self):
        for name,kind in [('apple','convex'),('tomato_soup_can','cylinder')]:
            s=geometry(name,kind);a=roll_state(s,1.,.1,.7)
            model=mujoco.MjModel.from_xml_string(s['xml']);data=mujoco.MjData(model)
            data.qpos[:]=np.r_[[0,0,1.],a['quat']];data.qvel[:]=a['velocity'];mujoco.mj_forward(model,data)
            velocity=np.zeros(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,model.body('object').id,velocity,0)
            np.testing.assert_allclose(velocity[3:]+np.cross(velocity[:3],a['support']),0.,atol=1e-10)

    def test_bounce_ballistic_energy_and_different_launches(self):
        speeds={}
        for name,p in catalog()['bounce'].items():
            s=geometry(name,'sphere',p['inertia_factor']);position,v,plan=bounce_launch(p,s['radius'],np.random.RandomState(0))
            self.assertAlmostEqual(p['normal_cor_target']**2*(position[2]-s['radius']+v[2]**2/(2*9.81)),plan['requested_apex_clearance_m'],places=10)
            self.assertTrue(np.isfinite(v).all());self.assertLessEqual(abs(v[2]),8.5);speeds[name]=abs(v[2])
        self.assertGreater(speeds['softball'],speeds['racquetball'])

    def test_allocation_and_invalid_weights(self):
        d=catalog()
        for motion,n in [('roll',4),('bounce',3)]:
            self.assertEqual({choose(motion,'default','mixed',None,d,i,n) for i in range(n)},set(d[motion]))
        with self.assertRaises(ValueError):choose('roll','default','mixed',{'apple':-1},d)
        with self.assertRaises(ValueError):choose('roll','default','fixed',None,d)

    def test_async_vector_has_distinct_objects(self):
        env=make_benchmark_vector(4,benchmark_backend='async',object_benchmark='roll_v1',object_motion='roll',task='Tracking',test_object='default',roll_ball_mode='mixed',viewer=False,render_mode=None,object_eval=False,benchmark_seed=123)
        try:
            self.assertEqual(set(env.call('benchmark_object')),{'orange','apple','lemon','tomato_soup_can'})
            obs,info=env.reset(seed=123);self.assertEqual(len(obs['hand']),4)
        finally:env.close()

    def test_shape_bounds_and_contact_remapping(self):
        env=make_object_benchmark(object_benchmark='roll_v1',test_object='ycb:tomato_soup_can',object_motion='roll',task='Catching',viewer=False,render_mode=None,object_eval=False)
        try:
            env.reset(seed=0);lo,hi=env._benchmark_world_bounds()
            self.assertAlmostEqual((hi[2]-lo[2])/2,env.benchmark_shape['radius'],places=6)
            self.assertGreater(env.benchmark_shape['marker_radius'],env.benchmark_shape['radius'])
            for _ in range(2):
                obs,r,t,tr,info=env.step({'base':np.zeros(2),'arm':np.zeros(3),'hand':np.zeros(12)})
                self.assertTrue(np.isfinite(r))
                self.assertFalse(np.any(np.isin(env.contacts['object_contacts'],[env.object_id,env.Dcmm.model.geom('benchmark_collision').id])))
        finally:env.close()

    def test_fruit_launch_uses_far_position_and_explicit_speed_override(self):
        for name in ('orange','apple','lemon'):
            env=make_object_benchmark(object_benchmark='roll_v1',test_object='ycb:'+name,
                object_motion='roll',task='Catching',viewer=False,render_mode=None,object_eval=False)
            try:
                env.reset(seed=0)
                launch=env.benchmark_episode['launch']
                distance=[2.,2.2] if name=='orange' else [1.9,2.1]
                self.assertEqual(launch['configured_distance_m'],distance)
                self.assertEqual(launch['configured_speed_mps'],catalog()['roll'][name]['speed_mps'])
                self.assertTrue(distance[0]<=env.object_pos3d[1]<=distance[1])
                speed=np.linalg.norm(env.object_vel6d[:2])
                self.assertTrue(launch['configured_speed_mps'][0]<=speed<=launch['configured_speed_mps'][1])
            finally:env.close()
        env=make_object_benchmark(object_benchmark='roll_v1',test_object='ycb:apple',
            object_motion='roll',task='Catching',viewer=False,render_mode=None,object_eval=False,roll_speed=.9)
        try:
            env.reset(seed=0)
            self.assertEqual(env.benchmark_episode['launch']['configured_speed_mps'],[.9,.9])
            self.assertAlmostEqual(np.linalg.norm(env.object_vel6d[:2]),.9)
        finally:env.close()

if __name__=='__main__':unittest.main()
