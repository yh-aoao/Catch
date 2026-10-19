"""Opt-in object evaluation factory; historical routes remain unchanged by default."""
import inspect
import numpy as np
import mujoco
from gym_dcmm.utils.test_objects import PRESETS, object_xml


def make_object_test(test_object, **kwargs):
    if test_object not in PRESETS:
        raise ValueError('Unknown test_object: ' + str(test_object))
    if kwargs.get('object_eval', False):
        raise ValueError('test_object requires object_eval=false')
    motion, task = kwargs.get('object_motion', 'throw'), kwargs.get('task', 'Tracking')
    if motion not in ('throw', 'roll', 'bounce'):
        raise ValueError('test_object supports throw/roll/bounce only')
    from gym_dcmm.envs.DcmmVecEnv import DcmmVecEnv, RollTrackingEnv, BounceEnv
    base = DcmmVecEnv
    if motion == 'roll':
        if task == 'Tracking':
            base = RollTrackingEnv
        else:
            from gym_dcmm.envs.DcmmVecEnv_roll_645edc4 import DcmmVecEnv as base
    if motion == 'bounce':
        if kwargs.get('bounce_physics', 'legacy') != 'legacy' or kwargs.get('bounce_launch', 'legacy') != 'legacy':
            raise ValueError('Historical bounce route requires legacy P/L')
        base = BounceEnv
        if task == 'Tracking':
            from gym_dcmm.envs.DcmmVecEnv_bounce_july22 import DcmmVecEnv as base
    method = {'throw': '_reset_object_throw', 'roll': '_reset_object_roll', 'bounce': '_reset_object_bounce'}[motion]
    # Actual throw helper name is resolved explicitly below.
    if motion == 'throw':
        method = '_reset_object_throw'
    original = getattr(base, method)
    def build(self):
        xml = object_xml(original(self), test_object)
        self.random_mass = PRESETS[test_object][2]
        return xml
    def reset_sim(self):
        base._reset_simulation(self)
        if not hasattr(self, '_test_object_constants'):
            return
        model, data = self.Dcmm.model, self.Dcmm.data
        gid = self.object_id
        bid = int(model.geom_bodyid[gid])
        for name, value in self._test_object_constants.items():
            getattr(model, name)[gid if name.startswith('geom_') else bid] = value
        self.random_mass = PRESETS[test_object][2]
        mujoco.mj_setConst(model, data)
        mujoco.mj_forward(model, data)
    def reset(self, *args, **kw):
        obs, info = base.reset(self, *args, **kw)
        info['test_object'] = test_object
        info['test_object_mass'] = self.random_mass
        return obs, info
    cls = type('ObjectTest' + base.__name__, (base,), {method: build, '_reset_simulation': reset_sim, 'reset': reset})
    accepted = inspect.signature(base.__init__).parameters
    env = cls(**{k: v for k, v in kwargs.items() if k in accepted})
    gid = env.object_id
    bid = int(env.Dcmm.model.geom_bodyid[gid])
    env._test_object_constants = {name: getattr(env.Dcmm.model, name)[gid if name.startswith('geom_') else bid].copy()
        for name in ('geom_size', 'body_mass', 'body_inertia', 'body_ipos', 'body_iquat')}
    print('[test-object] preset={} shape={} size={} mass={} environment={}'.format(
        test_object, *PRESETS[test_object], base.__module__ + '.' + base.__name__), flush=True)
    return env
