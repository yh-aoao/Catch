"""Opt-in object evaluation factory; historical routes remain unchanged by default."""
import inspect
import math
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from gym_dcmm.utils.test_objects import PRESETS, object_xml


def make_object_test(test_object, ycb_joint_damping=None, ycb_collision='auto', roll_speed=None,
                     roll_wait_guard=False, **kwargs):
    if roll_wait_guard and kwargs.get('object_motion', 'throw') != 'roll':
        raise ValueError('roll_wait_guard requires object_motion=roll')
    from gym_dcmm.utils.roll_speed import speed_range, override_velocity
    roll_speed = speed_range(roll_speed)
    if roll_speed is not None and kwargs.get('object_motion', 'throw') != 'roll':
        raise ValueError('roll_speed requires object_motion=roll')
    ycb = test_object.startswith('ycb:')
    if ycb_collision not in ('auto', 'mesh', 'sphere') or (not ycb and ycb_collision != 'auto'):
        raise ValueError('ycb_collision requires a YCB object and auto/mesh/sphere')
    if ycb_joint_damping is not None:
        if not ycb or not math.isfinite(float(ycb_joint_damping)) or float(ycb_joint_damping) < 0:
            raise ValueError('ycb_joint_damping requires a YCB object and a finite nonnegative value')
        ycb_joint_damping = float(ycb_joint_damping)
    resource = None
    if ycb:
        from ycb_objects.loader import load, compose, BALL_OBJECTS, sphere_radius
        resource = load(test_object[4:])
        if ycb_collision == 'auto':
            ycb_collision = 'sphere' if test_object[4:] in BALL_OBJECTS else 'mesh'
        if ycb_collision == 'sphere' and test_object[4:] not in BALL_OBJECTS:
            raise ValueError('Sphere collision is supported only for tennis_ball and baseball')
    preset = (('ycb_' + ycb_collision, (sphere_radius(resource) if ycb_collision == 'sphere' else resource[2],),
               resource[1]) if ycb else PRESETS.get(test_object))
    if preset is None:
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
        xml = compose(original(self), resource, collision=ycb_collision) if ycb else object_xml(original(self), test_object)
        if ycb_joint_damping is not None:
            root = ET.fromstring(xml)
            root.find('.//body[@name="object"]/joint').set('damping', str(ycb_joint_damping))
            xml = ET.tostring(root, encoding='unicode')
        self.random_mass = preset[2]
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
        self.random_mass = preset[2]
        if ycb:
            ids = np.flatnonzero((model.geom_bodyid == bid) & (np.arange(model.ngeom) != gid))
            for name in ('geom_friction', 'geom_solref', 'geom_solimp', 'geom_condim'):
                getattr(model, name)[ids] = getattr(model, name)[gid]
        mujoco.mj_setConst(model, data)
        mujoco.mj_forward(model, data)
    def reset(self, *args, **kw):
        obs, info = base.reset(self, *args, **kw)
        if roll_wait_guard:
            from gym_dcmm.utils.roll_wait_guard import RollWaitGuard
            self._roll_wait_guard = RollWaitGuard(self)
        info['test_object'] = test_object
        info['test_object_mass'] = self.random_mass
        if motion == 'roll':
            info['roll_initial_velocity'] = self.object_vel6d.copy()
            info['roll_initial_speed'] = float(np.linalg.norm(self.object_vel6d[:2]))
        if ycb:
            info['test_object_collision'] = ycb_collision
            model = self.Dcmm.model
            bid = int(model.geom_bodyid[self.object_id])
            dof = model.jnt_dofadr[model.body_jntadr[bid]]
            info['test_object_joint_damping'] = model.dof_damping[dof:dof+6].copy()
            info['test_object_env_time'] = float(self.env_time)
        return obs, info
    def contacts(self):
        result = base._get_contacts(self)
        if not ycb:
            return result
        model, data = self.Dcmm.model, self.Dcmm.data
        bid = int(model.geom_bodyid[self.object_id])
        ids = set(np.flatnonzero(model.geom_bodyid == bid).tolist())
        pairs = np.asarray(data.contact.geom).reshape(-1, 2)
        from ycb_objects.loader import remap_contacts
        return remap_contacts(result, pairs, ids, self.object_id)
    methods = {method: build, '_reset_simulation': reset_sim, 'reset': reset, '_get_contacts': contacts}
    if roll_wait_guard:
        def simulation_step(self, action):
            return base._step_mujoco_simulation(self, self._roll_wait_guard.filter(action))
        def step(self, action):
            obs, reward, terminated, truncated, info = base.step(self, action)
            info['roll_wait_guard'] = dict(self._roll_wait_guard.diagnostics)
            return obs, reward, terminated, truncated, info
        methods.update(_step_mujoco_simulation=simulation_step, step=step)
    if roll_speed is not None:
        def roll_pose(self):
            base.random_object_pose_roll(self)
            override_velocity(self, roll_speed)
        methods['random_object_pose_roll'] = roll_pose
    cls = type('ObjectTest' + base.__name__, (base,), methods)
    accepted = inspect.signature(base.__init__).parameters
    env = cls(**{k: v for k, v in kwargs.items() if k in accepted})
    # Historical roll_compat attaches this after construction; mirror its contract.
    env.roll_log = bool(kwargs.get('roll_log', False))
    gid = env.object_id
    bid = int(env.Dcmm.model.geom_bodyid[gid])
    env._test_object_constants = {name: getattr(env.Dcmm.model, name)[gid if name.startswith('geom_') else bid].copy()
        for name in ('geom_size', 'body_mass', 'body_inertia', 'body_ipos', 'body_iquat')}
    print('[test-object] preset={} shape={} size={} mass={} environment={}'.format(
        test_object, *preset, base.__module__ + '.' + base.__name__), flush=True)
    if roll_speed is not None:
        print('[roll-speed] initial speed range={} m/s'.format(roll_speed), flush=True)
    if roll_wait_guard:
        print('[roll-wait-guard] enabled: full-hand front clearance, 0.50m waiting height; release after ball clears table', flush=True)
    return env
