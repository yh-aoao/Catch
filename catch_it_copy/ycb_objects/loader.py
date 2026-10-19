"""Compose YCB collision meshes with a noncolliding legacy radius marker."""
from pathlib import Path
import copy, math
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]/'assets/ycb'

def load(name):
    if not name.replace('_','').isalnum(): raise ValueError('Invalid YCB name')
    path=ROOT/(name+'.xml')
    if not path.exists(): raise FileNotFoundError('Run python -m ycb_objects.download --objects '+name)
    root=ET.parse(path).getroot();body=root.find('./worldbody/body')
    mass=float(body.find('inertial').get('mass'))
    vertices=[]
    for mesh in root.findall('./asset/mesh'):
        f=ROOT/mesh.get('file')
        if f.suffix=='.obj':
            for line in f.read_text().splitlines():
                if line.startswith('v '): vertices.append(list(map(float,line.split()[1:4])))
    if not vertices: raise ValueError('No OBJ vertices for bounding radius')
    radius=max(math.sqrt(sum(x*x for x in v)) for v in vertices)
    return root,mass,radius

def compose(xml, resource):
    src,mass,radius=resource;root=ET.fromstring(xml);body=root.find(".//body[@name='object']")
    old=body.find("geom[@name='object']"); material_keys=('friction','solref','solimp','condim')
    contact={k:old.get(k) for k in material_keys if old.get(k) is not None}
    for node in list(body):
        if node.tag in ('geom','inertial'):body.remove(node)
    ET.SubElement(body,'geom',name='object',type='sphere',size=str(radius),mass='0',contype='0',conaffinity='0',rgba='0 0 0 0')
    asset=root.find('asset')
    if asset is None:asset=ET.SubElement(root,'asset')
    for node in src.findall('./asset/*'):
        node=copy.deepcopy(node)
        if node.get('file'):node.set('file',str((ROOT/node.get('file')).resolve()))
        asset.append(node)
    geoms=src.findall('./worldbody/body/geom')
    collisions=[g for g in geoms if g.get('contype','1')!='0' or g.get('conaffinity','1')!='0']
    if not collisions:raise ValueError('YCB has no collision pieces')
    for g in geoms:
        node=copy.deepcopy(g);is_collision=g in collisions
        node.set('mass',str(mass/len(collisions) if is_collision else 0.))
        if is_collision:node.attrib.update(contact)
        # Source angles are radians; convert explicit euler to quaternion before merging.
        e=node.attrib.pop('euler',None)
        if e and any(float(v)!=0 for v in e.split()):raise ValueError('Nonzero YCB geom Euler requires conversion')
        body.append(node)
    return ET.tostring(root,encoding='unicode')


def remap_contacts(result, pairs, object_ids, marker_id):
    """Keep robot counterpart IDs; normalize all object pieces to the legacy ID."""
    import numpy as np
    ids = set(map(int, object_ids))
    result = dict(result)
    result['object_contacts'] = np.asarray([int(b if a in ids else a)
        for a,b in np.asarray(pairs).reshape(-1,2) if (a in ids) != (b in ids)], dtype=int)
    for key in ('hand_contacts', 'base_contacts'):
        result[key] = np.asarray([marker_id if int(g) in ids else int(g) for g in result[key]], dtype=int)
    return result
