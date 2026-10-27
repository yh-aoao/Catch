import unittest
import mujoco
import numpy as np

from gym_dcmm.envs.roll_physical import make_roll_physical,load_materials
from ycb_objects.roll_physics_validation import trial


class RollPhysicalTests(unittest.TestCase):
    def test_runtime_routes_reset_dynamics_and_reward_integration(self):
        catalog=load_materials()['profiles']
        for task in ('Tracking','Catching'):
            for name in ('default','ycb:tennis_ball','ycb:baseball'):
                env=make_roll_physical(task=task,test_object=name,object_motion='roll',
                    viewer=False,render_mode=None,object_eval=False,roll_speed=.8,
                    roll_reward_version='adaptive_wait_v1')
                try:
                    for seed in (0,1):
                        _,info=env.reset(seed=seed)
                        b=env.unwrapped
                        model,data=b.Dcmm.model,b.Dcmm.data
                        bid=model.body('object').id
                        joint=model.body_jntadr[bid]
                        dof,q=model.jnt_dofadr[joint],model.jnt_qposadr[joint]
                        profile=catalog[info['roll_profile']]
                        np.testing.assert_allclose(model.dof_armature[dof:dof+6],0)
                        np.testing.assert_allclose(model.dof_damping[dof:dof+6],0)
                        np.testing.assert_allclose(model.body_ipos[bid],0)
                        np.testing.assert_allclose(model.body_inertia[bid],profile['inertia_factor']*profile['mass_kg']*profile['radius_m']**2)
                        np.testing.assert_allclose(data.qpos[q:q+3],b.object_pos3d)
                        np.testing.assert_allclose(data.qpos[15:21],b.Dcmm.data_arm.qpos[:6])
                        self.assertAlmostEqual(np.linalg.norm(b.object_vel6d[:2]),.8)
                        pair=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_PAIR,'roll_physical_table')
                        self.assertEqual(model.pair_dim[pair],6)
                        self.assertEqual(model.pair_friction[pair,3],profile['rolling_length_m'])
                        before=data.time
                        _,reward,_,_,info=env.step(dict(base=np.zeros(2),arm=np.zeros(6),hand=np.zeros(12)))
                        self.assertAlmostEqual(data.time-before,.04)
                        self.assertTrue(np.isfinite(reward))
                        self.assertTrue(np.isfinite(data.qpos).all())
                        self.assertEqual(info['roll_reward_version'],'adaptive_wait_v1')
                finally:
                    env.close()

    def test_passive_rolling_loss_and_step_refinement(self):
        for name,p in load_materials()['profiles'].items():
            with self.subTest(name=name):
                rolling=trial(p,extended_table=True,duration=2.)
                free=trial(p,extended_table=True,duration=2.,rolling_length=0.)
                self.assertLess(rolling['final_speed'],free['final_speed']-.02)
                self.assertGreater(free['final_speed'],.99)
                coarse,fine=trial(p),trial(p,timestep=.001)
                self.assertIsNotNone(coarse['exit'])
                self.assertLess(abs(coarse['exit']['speed']-fine['exit']['speed']),.03)
                self.assertLess(coarse['max_passive_energy_increase_j'],.001)

    def test_mixed_weights_repeatable(self):
        env=make_roll_physical(task='Catching',object_motion='roll',roll_ball_mode='mixed',
            roll_ball_weights=dict(original=.6,tennis_ball=.2,baseball=.2),viewer=False,render_mode=None,object_eval=False)
        try:
            names=[]
            for seed in range(30):
                env.reset(seed=seed)
                names.append(env.roll_physics_episode['profile'])
            self.assertEqual(set(names),{'original','tennis_ball','baseball'})
            env.reset(seed=0)
            self.assertEqual(env.roll_physics_episode['profile'],names[0])
        finally:
            env.close()


if __name__=='__main__':
    unittest.main()
