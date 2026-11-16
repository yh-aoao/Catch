"""Opt-in original five-shape Throw randomization with consistent solid inertia."""
import xml.etree.ElementTree as ET
import numpy as np

def randomize_xml(xml, config, rng=np.random):
    root=ET.fromstring(xml)
    body=root.find(".//body[@name='object']")
    if body is None:raise ValueError('Missing object body')
    geom=body.find("geom[@name='object']")
    if geom is None:raise ValueError('Missing object collision geom')
    shape=str(rng.choice(config.object_shape))
    size=[float(rng.uniform(lo,hi)) for lo,hi in config.object_size[shape]]
    mass=float(rng.uniform(*config.object_mass))
    damping=float(rng.uniform(*config.object_damping))
    geom.attrib.pop('mesh',None)
    geom.set('type',shape);geom.set('size',' '.join(map(str,size)))
    geom.set('mass',str(mass));geom.attrib.pop('density',None)
    # Let MuJoCo infer the uniform solid inertia for the sampled shape/size/mass.
    for child in list(body):
        if child.tag=='inertial':body.remove(child)
    others=[g for g in body.findall('geom') if g is not geom]
    if others:raise ValueError('Unexpected auxiliary object geoms; inertia needs explicit handling')
    joint=body.find('joint')
    if joint is not None:joint.set('damping',str(damping))
    return ET.tostring(root,encoding='unicode'),dict(version='multishape_v1',sampling='environment_creation',shape=shape,size=size,mass=mass,damping=damping)
