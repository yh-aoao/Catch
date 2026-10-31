"""Test-only YCB spheres for the fixed, physically prepared Basket policy.

The hand pose comes from the caller's configuration; this factory never chooses
a different pose automatically. Reset collision checks, gravity settling and
rewards are unchanged. Object dimensions, mass and sphere inertia use the existing
material priors; they are approximate properties, not measured YCB contacts.
"""
import inspect
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from gym_dcmm.envs.DcmmVecEnv import DcmmVecEnv
from ycb_objects.loader import BALL_OBJECTS, compose, load


PRIORS = Path(__file__).resolve().parents[2] / 'ycb_objects/calibration_data/balls.roll_table_priors_v1.json'


def sphere_xml(xml, resource, profile):
    """Keep one real collision sphere named object, plus noncolliding visuals."""
    root = ET.fromstring(compose(xml, resource, collision='sphere'))
    body = root.find(".//body[@name='object']")
    body.remove(body.find("geom[@name='object']"))
    sphere = body.find("geom[@name='object_collision_sphere']")
    radius, mass = profile['radius_m'], profile['mass_kg']
    sphere.set('name', 'object')
    sphere.set('size', str(radius))
    sphere.set('condim', '6')
    sphere.set('friction', '{} {} {}'.format(profile['sliding_friction'],
        profile['torsional_length_m'], profile['rolling_length_m']))
    # Dissipative contact for hand support, as in the original Basket protocol.
    sphere.set('solref', '0.02 1')
    sphere.set('solimp', '0.9 0.95 0.01')
    inertia = profile['inertia_factor'] * mass * radius * radius
    inertial = body.find('inertial')
    inertial.set('mass', str(mass))
    inertial.set('pos', '0 0 0')
    inertial.set('quat', '1 0 0 0')
    inertial.set('diaginertia', '{} {} {}'.format(inertia, inertia, inertia))
    joint = body.find('joint')
    joint.set('damping', '0')
    joint.set('armature', '0')
    return ET.tostring(root, encoding='unicode')


def make_basket_object_test(test_object, ycb_collision='auto',
                            ycb_joint_damping=None, **kwargs):
    """Construct a fixed Basket evaluation with actual YCB ball dimensions."""
    name = str(test_object)[4:] if str(test_object).startswith('ycb:') else ''
    if test_object != 'ycb:' + name or name not in BALL_OBJECTS:
        raise ValueError('Basket object test supports ycb:tennis_ball and ycb:baseball')
    if kwargs.get('object_motion') not in ('basket', 'throw_basket'):
        raise ValueError('Basket object test requires Basket motion')
    if (kwargs.get('object_eval', False) or kwargs.get('task') != 'Catching'
            or not kwargs.get('basket_fixed_base_training')
            or not kwargs.get('basket_fixed_physical')
            or kwargs.get('basket_reward_version') != 'basket_aim_v1'):
        raise ValueError('YCB Basket requires fixed-base basket_aim_v1 with physical preparation')
    if ycb_collision not in ('auto', 'sphere') or ycb_joint_damping not in (None, 0):
        raise ValueError('YCB Basket uses analytic sphere collision and zero free-joint damping')
    profile = json.loads(PRIORS.read_text())['profiles'][name]
    resource = load(name)

    class BasketObjectTest(DcmmVecEnv):
        def _reset_object_throw_basket(self):
            # basket_aim.make_config already made this an instance-local object.
            self.basket_cfg.basket_ball_radius = profile['radius_m']
            self.basket_cfg.basket_ball_mass = profile['mass_kg']
            xml = super()._reset_object_throw_basket()
            self.random_mass = profile['mass_kg']
            return sphere_xml(xml, resource, profile)

        def reset(self, *args, **kw):
            obs, info = super().reset(*args, **kw)
            info.update(test_object=test_object, test_object_collision='sphere',
                test_object_mass=profile['mass_kg'], test_object_radius=profile['radius_m'],
                test_object_material_status='simulation_priors_not_measured')
            return obs, info

    accepted = inspect.signature(DcmmVecEnv.__init__).parameters
    env = BasketObjectTest(**{k: v for k, v in kwargs.items() if k in accepted})
    env.basket_test_object = test_object
    print('[basket-test-object] {} mass={}kg radius={}m inertia_factor={} '
          'collision=sphere preparation=physical hand_pose=from_config'.format(
              test_object, profile['mass_kg'], profile['radius_m'],
              profile['inertia_factor']), flush=True)
    return env
