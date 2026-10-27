import unittest
from unittest.mock import patch
import mujoco
import numpy as np
from gym_dcmm.envs.bounce_physical_v2 import make_bounce_physical_v2, configuration


class PhysicalV2Tests(unittest.TestCase):
    def test_weighted_three_ball_sampling(self):
        from pathlib import Path
        from collections import Counter
        from gym_dcmm.envs.bounce_physical_v2 import sampling_probabilities
        names=['original','tennis_ball','baseball']
        weights=dict(original=.6,tennis_ball=.2,baseball=.2)
        np.testing.assert_allclose(sampling_probabilities(names,weights),[.6,.2,.2])
        for bad in ({'original':1},dict(original=0,tennis_ball=0,baseball=0),dict(original=-1,tennis_ball=1,baseball=1)):
            with self.assertRaises(ValueError):
                sampling_probabilities(names,bad)
        file=Path(__file__).resolve().parents[1]/'ycb_objects/calibration_data/balls.three_ball_training_v1.json'
        env=make_bounce_physical_v2(task='Catching',object_motion='bounce',object_eval=False,
            viewer=False,render_mode=None,bounce_ball_mode='mixed',bounce_ball_weights=weights,
            bounce_launch='baseball_intercept_v1',bounce_material_file=file)
        try:
            counts=Counter()
            for seed in range(100):
                env.reset(seed=seed)
                p=env.bounce_v2_episode
                counts[p['profile']]+=1
                self.assertEqual(p['launch_profile']=='baseball_intercept_v1',p['profile']=='baseball')
            self.assertEqual(set(counts),set(names))
            self.assertGreater(counts['original'],counts['tennis_ball'])
            self.assertGreater(counts['original'],counts['baseball'])
        finally:
            env.close()

    def test_baseball_launch_preset_is_selective_and_preserves_material(self):
        from pathlib import Path
        material_file=Path(__file__).resolve().parents[1]/'ycb_objects/calibration_data/balls.hard_floor_literature_v1.json'
        for task in ('Tracking','Catching'):
            for name in ('tennis_ball','baseball'):
                states=[]
                for mode in ('legacy','baseball_intercept_v1'):
                    env=make_bounce_physical_v2(task=task,test_object='ycb:'+name,object_motion='bounce',
                        bounce_launch=mode,bounce_material_file=material_file,
                        object_eval=False,viewer=False,render_mode=None)
                    try:
                        _,info=env.reset(seed=123)
                        pair=mujoco.mj_name2id(env.Dcmm.model,mujoco.mjtObj.mjOBJ_PAIR,'bounce_v2_ground')
                        states.append((env.object_pos3d.copy(),env.object_vel6d.copy(),
                            env.Dcmm.model.pair_solref[pair].copy(),env.Dcmm.model.pair_friction[pair].copy()))
                        if mode=='baseball_intercept_v1' and name=='baseball':
                            self.assertTrue(1.2<=env.object_pos3d[2]<=1.4)
                            self.assertTrue(1.5<=env.object_pos3d[1]<=1.8)
                            self.assertTrue(-3.<=env.object_vel6d[2]<=-2.5)
                            self.assertEqual(info['bounce_launch_profile'],'baseball_intercept_v1')
                    finally:
                        env.close()
                np.testing.assert_array_equal(states[0][2],states[1][2])
                np.testing.assert_array_equal(states[0][3],states[1][3])
                if name=='tennis_ball':
                    np.testing.assert_array_equal(states[0][0],states[1][0])
                    np.testing.assert_array_equal(states[0][1],states[1][1])

    def test_routes_reset_timing_and_consistent_inertia(self):
        for task in ('Tracking', 'Catching'):
            for name in ('default', 'ycb:tennis_ball', 'ycb:baseball'):
                env = make_bounce_physical_v2(task=task, test_object=name, object_motion='bounce',
                    object_eval=False, viewer=False, render_mode=None)
                try:
                    for _ in range(2):
                        _, info = env.reset(seed=123)
                        model, data = env.Dcmm.model, env.Dcmm.data
                        bid = model.body('object').id
                        j = model.body_jntadr[bid]
                        v, q = model.jnt_dofadr[j], model.jnt_qposadr[j]
                        np.testing.assert_allclose(data.qpos[q:q+3], env.object_pos3d)
                        np.testing.assert_allclose(data.qpos[15:21], env.Dcmm.data_arm.qpos[:6])
                        np.testing.assert_allclose(model.dof_armature[v:v+6], 0)
                        np.testing.assert_allclose(model.dof_damping[v:v+6], 0)
                        p = env.bounce_v2_episode
                        np.testing.assert_allclose(model.body_inertia[bid], p['inertia_factor']*p['mass']*p['radius']**2)
                        pair = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_PAIR, 'bounce_v2_ground')
                        self.assertGreaterEqual(pair, 0)
                        np.testing.assert_allclose(model.pair_friction[pair, :2], .3)
                        self.assertEqual(env.fps, 25)
                        self.assertLessEqual(info['bounce_initial_position'][2], 1.1)
                        self.assertLessEqual(info['bounce_initial_position'][1], 2.)
                        before = data.time
                        result = env.step(dict(base=np.zeros(2), arm=np.zeros(6), hand=np.zeros(12)))
                        self.assertAlmostEqual(data.time-before, .04, places=10)
                        self.assertTrue(np.isfinite(data.qpos).all())
                        self.assertTrue(np.isfinite(result[1]))
                finally:
                    env.close()

    def test_mixed_sampling_repeatable_and_reward_unchanged(self):
        env = make_bounce_physical_v2(task='Catching', object_motion='bounce',
            bounce_ball_mode='mixed', object_eval=False, viewer=False, render_mode=None)
        try:
            records = []
            for seed in range(30):
                env.reset(seed=seed)
                records.append(dict(env.bounce_v2_episode))
            self.assertEqual({p['profile'] for p in records}, {'original','tennis_ball','baseball'})
            env.reset(seed=0)
            self.assertEqual(env.bounce_v2_episode, records[0])
            with patch.object(type(env).__mro__[1], 'step', return_value=({},123.45,True,False,{})):
                self.assertEqual(env.step({})[1:4], (123.45,True,False))
        finally:
            env.close()

    def test_invalid_configuration(self):
        for options in (dict(height=[1.1,.9]), dict(speed=[-1,1]), dict(ground_friction=-1),
                        dict(damping_ratio=[0,.1]), dict(mass_jitter=.5), dict(typo=1)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                configuration(options)


if __name__ == '__main__':
    unittest.main()
