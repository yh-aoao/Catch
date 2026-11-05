"""Opt-in per-environment physical randomization; never mutates cached geometry."""
import copy
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
import numpy as np

VERSION = 'physical_dr_v1'

def scale_object_visuals(xml,scale):
    if scale==1.:return xml
    root=ET.fromstring(xml);body=root.find('.//body[@name="object"]')
    names=set()
    for g in body.findall('geom'):
        if not g.get('mesh'):continue
        names.add(g.get('mesh'))
        if g.get('pos'):g.set('pos',' '.join(map(str,np.fromstring(g.get('pos'),sep=' ')*scale)))
    for mesh in root.findall('./asset/mesh'):
        if mesh.get('name') in names:
            mesh.set('scale',' '.join(map(str,np.fromstring(mesh.get('scale','1 1 1'),sep=' ')*scale)))
    return ET.tostring(root,encoding='unicode')

def sample(motion, rng=np.random):
    return dict(version=VERSION, sampling='environment_creation',
                size_scale=float(rng.uniform(.95, 1.05)),
                mass_scale=float(rng.uniform(.9, 1.1)),
                friction_scale=float(rng.uniform(.85, 1.15)),
                rolling_scale=float(rng.uniform(.8, 1.2)) if motion=='roll' else 1.,
                damping_scale=float(rng.uniform(.9, 1.1)) if motion=='bounce' else 1.)

def material(profile, draw):
    p=copy.deepcopy(profile)
    p['sliding_friction']*=draw['friction_scale']
    if 'rolling_length_m' in p:p['rolling_length_m']*=draw['rolling_scale']
    if 'damping_ratio' in p:p['damping_ratio']*=draw['damping_scale']
    # Keep nominal COR only as a baseline label; it is NOT the sampled COR.
    # Launcher retains its baseline energy envelope, with actual rebound varying.
    return p

def scaled_geometry(shape, draw):
    out=copy.deepcopy(shape);s=draw['size_scale'];m=draw['mass_scale']
    out['mass']*=m
    for k in ('radius','marker_radius'):out[k]*=s
    out['points']*=s
    out['extents']=(np.asarray(out['extents'])*s).tolist()
    out['inertia']*=m*s*s
    root=ET.fromstring(out['xml'])
    for mesh in root.findall('./asset/mesh'):
        scale=np.fromstring(mesh.get('scale','1 1 1'),sep=' ')*s
        mesh.set('scale',' '.join(map(str,scale)))
        if mesh.get('refpos'):
            mesh.set('refpos',' '.join(map(str,np.fromstring(mesh.get('refpos'),sep=' ')*s)))
    body=root.find('./worldbody/body')
    for g in body.findall('geom'):
        if g.get('pos'):g.set('pos',' '.join(map(str,np.fromstring(g.get('pos'),sep=' ')*s)))
        if g.get('size'):g.set('size',' '.join(map(str,np.fromstring(g.get('size'),sep=' ')*s)))
    inertial=body.find('inertial')
    inertial.set('mass',str(out['mass']))
    inertial.set('diaginertia',' '.join(map(str,out['inertia'])))
    out['xml']=ET.tostring(root,encoding='unicode')
    return out

def legacy_environment(factory, parameters, path, name, motion, draw):
    data=json.loads(Path(path).read_text());p=material(data['profiles'][name],draw)
    if motion=='bounce':
        from gym_dcmm.envs.bounce_physical_v2 import profiles
        mass,radius,_=profiles()[name]
        p['mass_kg']=(p.get('mass_kg') or mass)*draw['mass_scale']
        p['radius_m']=(p.get('radius_m') or radius)*draw['size_scale']
    else:
        p['mass_kg']*=draw['mass_scale'];p['radius_m']*=draw['size_scale']
    data['profiles'][name]=p
    # Factories eagerly read/capture this catalog; no file is needed on reset.
    with tempfile.TemporaryDirectory(prefix='catch_dr_') as temp:
        file=Path(temp)/'materials.json';file.write_text(json.dumps(data))
        parameters=dict(parameters,benchmark_visual_scale=draw['size_scale'],**{motion+'_material_file':str(file)})
        env=factory(**parameters)
    return env
