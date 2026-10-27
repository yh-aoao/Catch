"""Opt-in Roll sphere/material correction, preserving historical reward/control."""
import hashlib
import inspect
import json
from pathlib import Path
import random
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from gym_dcmm.envs.object_test import set_constants_preserving_state
from gym_dcmm.utils.roll_speed import speed_range, override_velocity

VERSION = 'physical_v1'
DEFAULT_MATERIALS = Path(__file__).resolve().parents[2]/'ycb_objects/calibration_data/balls.roll_table_priors_v1.json'


def load_materials(path=None):
    path=Path(path or DEFAULT_MATERIALS).resolve()
    raw=path.read_bytes()
    data=json.loads(raw)
    if data.get('schema')!='roll_table_materials_v1' or not data.get('surface_id'):
        raise ValueError('Roll requires roll_table_materials_v1 with surface_id')
    from ycb_objects.materials import finite
    profiles=data.get('profiles',{})
    if not profiles or set(profiles)-{'original','tennis_ball','baseball'}:
        raise ValueError('Invalid Roll ball profiles')
    for name,p in profiles.items():
        for key,limits in dict(mass_kg=(.001,1.),radius_m=(.005,.15),inertia_factor=(.1,2/3),
                               sliding_friction=(0.,3.),torsional_length_m=(0.,.005),rolling_length_m=(0.,.005)).items():
            finite(p.get(key),name+'.'+key,*limits)
    return dict(data,path=str(path),sha256=hashlib.sha256(raw).hexdigest())


def table_pair(xml,profile,resource=None):
    if resource is not None:
        from ycb_objects.loader import compose
        xml=compose(xml,resource,collision='sphere')
    root=ET.fromstring(xml)
    body=root.find('.//body[@name="object"]')
    p=profile
    mass,radius=p['mass_kg'],p['radius_m']
    inertial=body.find('inertial')
    if inertial is None:
        inertial=ET.SubElement(body,'inertial')
    inertial.attrib.update(mass=str(mass),pos='0 0 0',quat='1 0 0 0',
                          diaginertia=' '.join([str(p['inertia_factor']*mass*radius**2)]*3))
    body.find('joint').attrib.update(armature='0',damping='0')
    collision='object_collision_sphere' if resource is not None else 'object'
    for geom in body.findall('geom'):
        geom.set('mass','0')
        if geom.get('name') in ('object',collision):
            geom.set('size',str(radius))
        geom.set('solref','.02 1')
    contact=root.find('contact')
    if contact is None:
        contact=ET.SubElement(root,'contact')
    mu,tau,rho=p['sliding_friction'],p['torsional_length_m'],p['rolling_length_m']
    ET.SubElement(contact,'pair',name='roll_physical_table',geom1='table_surface',geom2=collision,
                  condim='6',friction=f'{mu} {mu} {tau} {rho} {rho}',
                  solref='.02 1',solimp='.9 .95 .001 .5 2')
    return ET.tostring(root,encoding='unicode')


def make_roll_physical(roll_physics=VERSION,roll_material_file=None,roll_ball_mode='fixed',
                       roll_ball_weights=None,roll_reward_version='legacy',roll_adaptive_wait=None,
                       test_object='default',roll_speed=None,roll_wait_guard=False,
                       ycb_collision='auto',ycb_joint_damping=None,**kwargs):
    if roll_physics!=VERSION or kwargs.get('object_motion')!='roll' or kwargs.get('object_eval',False):
        raise ValueError('physical_v1 requires object_motion=roll, object_eval=false')
    if roll_wait_guard or ycb_collision=='mesh' or ycb_joint_damping is not None:
        raise ValueError('physical_v1 owns sphere dynamics; disable hard guard/mesh/joint damping overrides')
    if roll_ball_mode not in ('fixed','mixed') or roll_reward_version not in ('legacy','adaptive_wait_v1'):
        raise ValueError('Invalid Roll ball mode or reward version')
    if roll_ball_mode=='mixed' and test_object!='default':
        raise ValueError('Roll mixed physics training requires test_object=default')
    if roll_ball_weights is not None and roll_ball_mode!='mixed':
        raise ValueError('roll_ball_weights requires mixed mode')
    materials=load_materials(roll_material_file)
    catalog=materials['profiles']
    name='original' if test_object in ('default','sphere') else (test_object[4:] if test_object.startswith('ycb:') else test_object)
    if name not in catalog:
        raise ValueError('Material profile missing: '+name)
    names=list(catalog)
    probabilities=None
    if roll_ball_weights is not None:
        if not isinstance(roll_ball_weights,dict) or set(roll_ball_weights)!=set(names):
            raise ValueError('roll_ball_weights must match material profile names')
        values=np.array([roll_ball_weights[n] for n in names],dtype=float)
        if not np.isfinite(values).all() or (values<0).any() or values.sum()<=0:
            raise ValueError('Invalid Roll sampling weights')
        probabilities=values/values.sum()
    bounds=speed_range(roll_speed)
    resource=None
    if test_object.startswith('ycb:'):
        from ycb_objects.loader import load
        resource=load(name)
    if kwargs.get('task')=='Tracking':
        from gym_dcmm.envs.DcmmVecEnv import RollTrackingEnv as Base
    else:
        from gym_dcmm.envs.DcmmVecEnv_roll_645edc4 import DcmmVecEnv as Base

    class PhysicalRoll(Base):
        def _reset_object_roll(self):
            return table_pair(super()._reset_object_roll(),catalog[name],resource)

        def _reset_simulation(self):
            super()._reset_simulation()
            selected=str(np.random.choice(names,p=probabilities)) if roll_ball_mode=='mixed' else name
            p=catalog[selected]
            mass,radius=p['mass_kg'],p['radius_m']
            model,data=self.Dcmm.model,self.Dcmm.data
            body=model.body('object').id
            dof=model.jnt_dofadr[model.body_jntadr[body]]
            collision=model.geom('object_collision_sphere').id if resource is not None else self.object_id
            model.body_mass[body]=mass
            model.body_inertia[body]=p['inertia_factor']*mass*radius**2
            model.body_ipos[body]=0
            model.body_iquat[body]=[1,0,0,0]
            model.geom_size[[self.object_id,collision],0]=radius
            model.geom_rbound[[self.object_id,collision]]=radius
            model.dof_armature[dof:dof+6]=0
            model.dof_damping[dof:dof+6]=0
            pair=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_PAIR,'roll_physical_table')
            mu,tau,rho=p['sliding_friction'],p['torsional_length_m'],p['rolling_length_m']
            model.pair_friction[pair]=[mu,mu,tau,rho,rho]
            set_constants_preserving_state(model,data)
            table=model.geom('table_surface').id
            top=float(data.geom_xpos[table,2]+model.geom_size[table,2])
            self.object_pos3d[2]=top+radius+.0002
            speed=float(np.linalg.norm(self.object_vel6d[:2]))
            override_velocity(self,bounds or (speed,speed))
            self.random_mass=mass
            self.Dcmm.set_throw_pos_vel(pose=np.r_[self.object_pos3d,self.object_q],velocity=self.object_vel6d)
            mujoco.mj_forward(model,data)
            self.roll_physics_episode=dict(profile=selected,version=VERSION,mass_kg=float(mass),
                radius_m=float(radius),inertia_factor=p['inertia_factor'],sliding_friction=mu,
                rolling_length_m=rho,torsional_length_m=tau,status=materials.get('status','simulation_prior'))

        def _get_contacts(self):
            result=super()._get_contacts()
            if resource is not None:
                from ycb_objects.loader import remap_contacts
                m=self.Dcmm.model
                ids=np.flatnonzero(m.geom_bodyid==m.body('object').id)
                result=remap_contacts(result,np.asarray(self.Dcmm.data.contact.geom).reshape(-1,2),ids,self.object_id)
            return result

        def reset(self,*,seed=None,options=None):
            if options:
                raise ValueError('Roll physical reset options unsupported')
            if seed is not None:
                np.random.seed(seed)
                random.seed(seed)
            obs,info=super().reset()
            info.update(roll_physics_version=VERSION,roll_profile=self.roll_physics_episode['profile'],
                        roll_initial_velocity=self.object_vel6d.copy(),roll_initial_position=self.object_pos3d.copy())
            return obs,info

        def step(self,action):
            obs,reward,term,trunc,info=super().step(action)
            info.update(roll_physics_version=VERSION,roll_profile=self.roll_physics_episode['profile'])
            return obs,reward,term,trunc,info

    accepted=inspect.signature(Base.__init__).parameters
    env=PhysicalRoll(**{k:v for k,v in kwargs.items() if k in accepted})
    env.roll_log=bool(kwargs.get('roll_log',False))
    if roll_reward_version=='adaptive_wait_v1':
        from gym_dcmm.envs.roll_adaptive import AdaptiveRollReward
        env=AdaptiveRollReward(env,roll_adaptive_wait)
    return env


def write_provenance(output,config):
    root=Path(__file__).resolve().parents[2]
    names=['train_DCMM.py','configs/config.yaml','gym_dcmm/envs/roll_physical.py',
           'gym_dcmm/envs/roll_adaptive.py','gym_dcmm/utils/roll_adaptive_wait.py',
           'gym_dcmm/envs/DcmmVecEnv_roll_645edc4.py','gym_dcmm/envs/DcmmVecEnv.py']
    result=dict(version=VERSION,materials=load_materials(config.get('roll_material_file')),
                ball_mode=config.get('roll_ball_mode'),weights=config.get('roll_ball_weights'),
                speed=config.get('roll_speed'),reward_version=config.get('roll_reward_version'),
                sources={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names})
    (Path(output)/'roll_physics_provenance.json').write_text(json.dumps(result,indent=2))
