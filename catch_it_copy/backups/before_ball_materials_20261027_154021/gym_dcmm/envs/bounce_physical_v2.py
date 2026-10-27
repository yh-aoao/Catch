"""Versioned spherical dynamics and a nearer launch for interception training.

Profiles are simulation priors, not measured YCB material calibration. Rewards,
action semantics and the 2 ms controller / 40 ms policy intervals are unchanged.
"""
import hashlib
import inspect
import json
import math
from pathlib import Path
import random
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from gym_dcmm.envs.object_test import set_constants_preserving_state

VERSION = 'physical_v2'
DT = .000125
SUBSTEPS = 16
DEFAULTS = dict(height=[.9, 1.1], distance=[1.6, 2.0], speed=[1.3, 1.7],
                vertical_speed=[-.2, -.2], heading_degrees=[-15., 15.],
                spin_factor=[.5, 1.5], damping_ratio=[.11, .11],
                ground_friction=.3, time_constant=.008,
                mass_jitter=.10, radius_jitter=.05)


def configuration(options=None):
    cfg = dict(DEFAULTS, **(options or {}))
    if set(cfg) != set(DEFAULTS):
        raise ValueError('Unknown bounce_v2 options: '+str(set(cfg)-set(DEFAULTS)))
    for key in ('height', 'distance', 'speed', 'vertical_speed', 'heading_degrees', 'spin_factor', 'damping_ratio'):
        a = np.asarray(cfg[key], dtype=float)
        if a.shape != (2,) or not np.isfinite(a).all() or a[0] > a[1]:
            raise ValueError('Invalid bounce_v2 range: '+key)
        cfg[key] = a.tolist()
    for key in ('height', 'distance', 'speed', 'damping_ratio'):
        if cfg[key][0] <= 0:
            raise ValueError(key+' must be positive')
    if not .5 <= cfg['height'][0] <= cfg['height'][1] <= 2.5:
        raise ValueError('Supported launch height envelope is 0.5–2.5 m; custom ranges require validation')
    if not .06 <= cfg['damping_ratio'][0] <= cfg['damping_ratio'][1] <= .25:
        raise ValueError('Supported damping-ratio envelope is .06–.25; custom ranges require validation')
    for key in ('ground_friction', 'time_constant', 'mass_jitter', 'radius_jitter'):
        if not np.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError('Invalid bounce_v2 scalar: '+key)
    if cfg['time_constant'] < .008 or max(cfg['mass_jitter'], cfg['radius_jitter']) > .25:
        raise ValueError('Use time_constant>=.008 and relative jitters<=.25')
    return cfg


def profiles():
    from ycb_objects.loader import load, sphere_radius
    result = dict(original=(.05, .04, .4))
    for name, inertia_factor in (('tennis_ball', 2/3), ('baseball', .4)):
        resource = load(name)
        result[name] = (resource[1], sphere_radius(resource), inertia_factor)
    return result


def _xml(xml, name, cfg, resource=None):
    if resource is not None:
        from ycb_objects.loader import compose
        xml = compose(xml, resource, collision='sphere')
    root = ET.fromstring(xml)
    body = root.find('.//body[@name="object"]')
    mass, radius, factor = profiles()[name]
    inertia = body.find('inertial')
    if inertia is None:
        inertia = ET.SubElement(body, 'inertial')
    inertia.attrib.update(mass=str(mass), pos='0 0 0', quat='1 0 0 0',
                          diaginertia=' '.join([str(factor*mass*radius**2)]*3))
    body.find('joint').attrib.update(damping='0', armature='0')
    collision_name = 'object_collision_sphere' if resource is not None else 'object'
    # Critical damping for ball/hand; explicit pair controls only ball/floor.
    for geom in body.findall('geom'):
        geom.set('mass', '0')
        if geom.get('name') in ('object', collision_name):
            geom.set('size', str(radius))
        geom.set('solref', '.02 1')
    contact = root.find('contact')
    if contact is None:
        contact = ET.SubElement(root, 'contact')
    mu = cfg['ground_friction']
    ET.SubElement(contact, 'pair', name='bounce_v2_ground', geom1='floor', geom2=collision_name,
                  condim='4', friction=f'{mu} {mu} .005 0 0',
                  solref=f"{cfg['time_constant']} {cfg['damping_ratio'][0]}",
                  solimp='.95 .998 .002 .5 2')
    return ET.tostring(root, encoding='unicode')


def make_bounce_physical_v2(test_object='default', bounce_physics=VERSION,
                            bounce_launch='legacy', bounce_ball_mode='fixed',
                            bounce_v2=None, **kwargs):
    if bounce_physics != VERSION or kwargs.get('object_motion') != 'bounce':
        raise ValueError('physical_v2 requires object_motion=bounce')
    if bounce_launch != 'legacy' or kwargs.get('object_eval', False):
        raise ValueError('physical_v2 owns its launch; use bounce_launch=legacy, object_eval=false')
    if bounce_ball_mode not in ('fixed', 'mixed'):
        raise ValueError('bounce_ball_mode must be fixed or mixed')
    name = 'original' if test_object in ('default', 'sphere') else (test_object[4:] if test_object.startswith('ycb:') else test_object)
    if name not in ('original', 'tennis_ball', 'baseball'):
        raise ValueError('physical_v2 currently supports original/tennis_ball/baseball spheres')
    if bounce_ball_mode == 'mixed' and test_object != 'default':
        raise ValueError('Mixed physics training uses test_object=default; YCB visual tests are fixed')
    if kwargs.get('ycb_collision', 'auto') == 'mesh' or kwargs.get('ycb_joint_damping') is not None:
        raise ValueError('physical_v2 requires analytical spheres and owns joint damping')
    cfg, catalog = configuration(bounce_v2), profiles()
    resource = None
    if test_object.startswith('ycb:'):
        from ycb_objects.loader import load
        resource = load(name)
    if kwargs.get('task') == 'Tracking':
        from gym_dcmm.envs.DcmmVecEnv_bounce_july22 import DcmmVecEnv as Base
    else:
        from gym_dcmm.envs.DcmmVecEnv import BounceEnv as Base

    class PhysicalBounce(Base):
        def _reset_object_bounce(self):
            return _xml(super()._reset_object_bounce(), name, cfg, resource)

        def _reset_simulation(self):
            super()._reset_simulation()
            model, data = self.Dcmm.model, self.Dcmm.data
            selected = str(np.random.choice(list(catalog))) if bounce_ball_mode == 'mixed' else name
            mass, radius, factor = catalog[selected]
            if bounce_ball_mode == 'mixed':
                mass *= np.random.uniform(1-cfg['mass_jitter'], 1+cfg['mass_jitter'])
                radius *= np.random.uniform(1-cfg['radius_jitter'], 1+cfg['radius_jitter'])
            bid = model.body('object').id
            dof = model.jnt_dofadr[model.body_jntadr[bid]]
            ids = np.flatnonzero(model.geom_bodyid == bid)
            collision = model.geom('object_collision_sphere').id if resource is not None else self.object_id
            model.body_mass[bid] = mass
            model.body_inertia[bid] = factor*mass*radius**2
            model.body_ipos[bid] = 0
            model.body_iquat[bid] = [1, 0, 0, 0]
            model.geom_size[[self.object_id, collision], 0] = radius
            model.geom_rbound[[self.object_id, collision]] = radius
            model.dof_armature[dof:dof+6] = 0
            model.dof_damping[dof:dof+6] = 0
            model.geom_solref[ids] = [.02, 1.]
            model.geom_solref[self.floor_id] = [.02, 1.]
            pair = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_PAIR, 'bounce_v2_ground')
            ratio = np.random.uniform(*cfg['damping_ratio'])
            model.pair_solref[pair] = [cfg['time_constant'], ratio]
            # Elliptic cone uses solref; the pyramidal model ignores solreffriction.
            model.pair_solreffriction[pair] = [cfg['time_constant'], ratio]
            model.opt.timestep = DT
            model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
            self._bounce_physics_substeps = SUBSTEPS
            self.fps = 1/(self.steps_per_policy*DT*SUBSTEPS)
            self.random_mass = mass
            set_constants_preserving_state(model, data)
            heading = math.radians(np.random.uniform(*cfg['heading_degrees']))
            speed = np.random.uniform(*cfg['speed'])
            vx, vy = speed*math.sin(heading), -speed*math.cos(heading)
            spin = np.random.uniform(*cfg['spin_factor'])
            self.object_pos3d = np.array([np.random.uniform(-.25, .25),
                np.random.uniform(*cfg['distance']), np.random.uniform(*cfg['height'])])
            self.object_q = np.array([1., 0., 0., 0.])
            self.object_vel6d = np.array([vx, vy, np.random.uniform(*cfg['vertical_speed']),
                spin*(-vy)/radius, spin*vx/radius, np.random.uniform(-3., 3.)])
            self.object_static_time = 0.
            self.Dcmm.set_throw_pos_vel(pose=np.r_[self.object_pos3d, self.object_q], velocity=self.object_vel6d)
            mujoco.mj_forward(model, data)
            self.bounce_v2_episode = dict(profile=selected, mass=float(mass), radius=float(radius),
                inertia_factor=float(factor), damping_ratio=float(ratio), material_status='uncalibrated_simulation_prior')

        def _get_contacts(self):
            result = super()._get_contacts()
            if resource is not None:
                from ycb_objects.loader import remap_contacts
                model = self.Dcmm.model
                ids = np.flatnonzero(model.geom_bodyid == model.body('object').id)
                result = remap_contacts(result, np.asarray(self.Dcmm.data.contact.geom).reshape(-1, 2), ids, self.object_id)
            return result

        def reset(self, *, seed=None, options=None):
            if options:
                raise ValueError('physical_v2 reset options are not supported')
            if seed is not None:
                np.random.seed(seed)
                random.seed(seed)
            obs, info = super().reset()
            info.update(bounce_physics_version=VERSION, bounce_profile=self.bounce_v2_episode['profile'],
                        bounce_mass=self.random_mass, bounce_radius=float(self.Dcmm.model.geom_size[self.object_id, 0]),
                        bounce_initial_position=self.object_pos3d.copy(), bounce_initial_velocity=self.object_vel6d.copy())
            return obs, info

        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            info.update(bounce_physics_version=VERSION, bounce_profile=self.bounce_v2_episode['profile'])
            return obs, reward, terminated, truncated, info

    accepted = inspect.signature(Base.__init__).parameters
    env = PhysicalBounce(**{key: value for key, value in kwargs.items() if key in accepted})
    env.bounce_v2_options = cfg
    return env


def write_provenance(output, config):
    root = Path(__file__).resolve().parents[2]
    names = ['train_DCMM.py', 'configs/config.yaml', 'gym_dcmm/envs/bounce_physical_v2.py',
             'gym_dcmm/envs/DcmmVecEnv.py', 'gym_dcmm/envs/DcmmVecEnv_bounce_july22.py']
    result = dict(version=VERSION, timestep=DT, controller_interval=DT*SUBSTEPS,
                  material_status='uncalibrated_simulation_prior', options=configuration(config.get('bounce_v2')),
                  ball_mode=config.get('bounce_ball_mode'), profiles=profiles(),
                  sources={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in names})
    (Path(output)/'bounce_physics_provenance.json').write_text(json.dumps(result, indent=2))
