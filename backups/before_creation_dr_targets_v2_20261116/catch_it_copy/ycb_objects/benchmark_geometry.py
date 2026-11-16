"""Rigid benchmark geometry: analytical balls/cans, convex fruit, no magic inertia."""
import copy
from functools import lru_cache
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from ycb_objects.loader import load, compose


def quat(matrix):
    q=Rotation.from_matrix(matrix).as_quat()
    return np.r_[q[3],q[:3]]


def align(source,target):
    a=np.asarray(source,dtype=float);a=a/np.linalg.norm(a)
    b=np.asarray(target,dtype=float);b=b/np.linalg.norm(b)
    v=np.cross(a,b);c=float(a@b)
    if c < -1+1e-9:
        axis=np.cross(a,[1.,0,0] if abs(a[0])<.9 else [0.,1,0]);axis/=np.linalg.norm(axis)
        return Rotation.from_rotvec(np.pi*axis).as_matrix()
    k=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3)+k+(k@k)/max(1+c,1e-12)


def mesh_points(model,data,geom):
    mesh=model.geom_dataid[geom];start=model.mesh_vertadr[mesh];count=model.mesh_vertnum[mesh]
    return np.asarray(model.mesh_vert[start:start+count],dtype=float)@data.geom_xmat[geom].reshape(3,3).T+data.geom_xpos[geom]


@lru_cache(maxsize=16)
def geometry(name,kind,factor=.4):
    resource=load(name)
    root=ET.fromstring(compose('<mujoco><worldbody><body name="object"><freejoint/><geom name="object" type="sphere" size=".04"/></body></worldbody></mujoco>',resource,collision='mesh'))
    body=root.find('.//body[@name="object"]')
    mass=resource[1]
    visual=next(g for g in body.findall('geom') if g.get('contype')=='0' and g.get('type')=='mesh')
    # Use the detailed visual hull for rounded fruit; the upstream five 32-face
    # pieces are unnecessarily faceted for rolling. MuJoCo collision is convex.
    for node in list(body):
        if node.tag=='inertial' or (node.tag=='geom' and node is not visual):body.remove(node)
    visual.set('mass','0')
    physical=copy.deepcopy(visual);physical.attrib.pop('material',None)
    physical.attrib.update(name='benchmark_collision',contype='1',conaffinity='1',group='3',rgba='0 0 0 0',density='1000')
    physical.attrib.pop('mass',None);body.append(physical)
    reference=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));data=mujoco.MjData(reference);mujoco.mj_forward(reference,data)
    gid=reference.geom('benchmark_collision').id
    points=mesh_points(reference,data,gid)
    center=reference.body_ipos[reference.body('object').id].copy()
    centered=points-center
    values,vectors=np.linalg.eigh(np.cov(centered.T));long_axis=vectors[:,-1]
    if kind=='sphere':
        radius=float(np.ptp(points,axis=0).mean()/2)
        center=(points.min(axis=0)+points.max(axis=0))/2
        transform=np.eye(3);diagonal=np.full(3,factor*mass*radius**2);inertia_quat=[1,0,0,0]
        support=np.array([[0.,0.,0.]])
        physical.attrib.clear();physical.attrib.update(name='benchmark_collision',type='sphere',size=str(radius),mass='0',contype='1',conaffinity='1',group='3',rgba='0 0 0 0')
        extents=[radius*2]*3
    elif kind=='cylinder':
        transform=align(long_axis,[0,0,1]);canonical=centered@transform.T
        # Fit cylinder to the source can's principal axis and radial extent.
        radius=float(np.percentile(np.linalg.norm(canonical[:,:2],axis=1),99.5))
        half_length=float(np.ptp(canonical[:,2])/2)
        center=center+long_axis*float((canonical[:,2].max()+canonical[:,2].min())/2)
        diagonal=np.array([mass*(3*radius**2+(2*half_length)**2)/12]*2+[.5*mass*radius**2])
        inertia_quat=[1,0,0,0]
        angles=np.linspace(0,2*np.pi,128,endpoint=False)
        support=np.array([[radius*np.cos(a),radius*np.sin(a),z] for z in (-half_length,half_length) for a in angles])
        physical.attrib.clear();physical.attrib.update(name='benchmark_collision',type='cylinder',size=f'{radius} {half_length}',mass='0',contype='1',conaffinity='1',group='3',rgba='0 0 0 0')
        extents=[2*radius,2*radius,2*half_length]
    elif kind=='convex':
        transform=np.eye(3);radius=float(np.max(np.linalg.norm(centered,axis=1)))
        bid=reference.body('object').id
        diagonal=reference.body_inertia[bid]*mass/reference.body_mass[bid]
        inertia_quat=reference.body_iquat[bid].tolist();support=centered
        physical.set('mass','0');physical.attrib.pop('density',None)
        extents=np.ptp(centered,axis=0).tolist()
    else:raise ValueError('Unsupported benchmark geometry: '+kind)
    # Recenter mesh frame at the estimated homogeneous COM; keep visuals and
    # collision aligned. Sphere/cylinder collision is already canonical.
    for g in (visual,physical) if kind=='convex' else (visual,):
        old=np.fromstring(g.get('pos','0 0 0'),sep=' ')
        g.set('pos',' '.join(map(str,transform@(old-center))))
        g.attrib.pop('euler',None);g.set('quat',' '.join(map(str,quat(transform))))
    ET.SubElement(body,'inertial',mass=str(mass),pos='0 0 0',quat=' '.join(map(str,inertia_quat)),diaginertia=' '.join(map(str,diagonal)))
    marker_radius=radius if kind=='sphere' else float(np.max(np.linalg.norm(support,axis=1)))
    ET.SubElement(body,'geom',name='object',type='sphere',size=str(marker_radius),mass='0',contype='0',conaffinity='0',rgba='0 0 0 0',group='5')
    return dict(xml=ET.tostring(root,encoding='unicode'),kind=kind,mass=mass,radius=radius,
                marker_radius=marker_radius,points=support,long_axis=transform@long_axis,
                inertia=diagonal,extents=extents,com_source=center.tolist(),
                inertia_basis='homogeneous visual convex hull' if kind=='convex' else 'structural analytical approximation')


def inject(xml,shape,profile,surface):
    root=ET.fromstring(xml);source=ET.fromstring(shape['xml']);body=root.find('.//body[@name="object"]')
    for node in list(body):
        if node.tag in ('geom','inertial'):body.remove(node)
    asset=root.find('asset')
    if asset is None:asset=ET.SubElement(root,'asset')
    for node in source.findall('./asset/*'):asset.append(copy.deepcopy(node))
    for node in source.findall('./worldbody/body/*'):
        if node.tag in ('geom','inertial'):body.append(copy.deepcopy(node))
    joint=body.find('joint')
    if joint is None:
        joint=body.find('freejoint')
        joint.tag='joint';joint.set('type','free')
    joint.attrib.update(damping='0',armature='0')
    for g in body.findall('geom'):
        g.set('solref','.02 1');g.set('solimp','.9 .95 .001 .5 2')
        g.set('friction',f"{profile['sliding_friction']} {profile['torsional_length_m']} {profile.get('rolling_length_m',0.)}")
    contact=root.find('contact')
    if contact is None:contact=ET.SubElement(root,'contact')
    bounce=surface=='floor'
    mu=profile['sliding_friction'];tau=profile['torsional_length_m'];rho=profile.get('rolling_length_m',0.)
    ET.SubElement(contact,'pair',name='benchmark_surface',geom1=surface,geom2='benchmark_collision',condim='4' if bounce else '6',
                  friction=f'{mu} {mu} {tau} {rho} {rho}',
                  solref=f"{profile.get('time_constant',.02)} {profile.get('damping_ratio',1.)}",solimp='.95 .998 .002 .5 2' if bounce else '.9 .95 .001 .5 2')
    return ET.tostring(root,encoding='unicode')


def bounds(shape,rotation):
    if shape['kind']=='sphere':return np.full(3,-shape['radius']),np.full(3,shape['radius'])
    if shape['kind']=='cylinder':
        r=shape['radius'];h=shape['extents'][2]/2;axis=rotation[:,2]
        extent=h*np.abs(axis)+r*np.sqrt(np.maximum(0,1-axis**2))
        return -extent,extent
    world=shape['points']@rotation.T
    return world.min(axis=0),world.max(axis=0)


def roll_state(shape,speed,heading,yaw=0.):
    velocity=np.array([speed*np.sin(heading),-speed*np.cos(heading),0.])
    axis=np.array([-velocity[1],velocity[0],0.])
    axis=axis/np.linalg.norm(axis) if np.linalg.norm(axis)>1e-12 else np.array([1.,0.,0.])
    if shape['kind']=='cylinder':rotation=align([0,0,1],axis)@Rotation.from_rotvec([0,0,yaw]).as_matrix()
    elif np.max(shape['extents'])/np.min(shape['extents'])>1.3:
        rotation=align(shape['long_axis'],axis)@Rotation.from_rotvec(shape['long_axis']*yaw).as_matrix()
    else:rotation=Rotation.from_euler('xyz',[0.,0.,yaw]).as_matrix()
    low,high=bounds(shape,rotation)
    if shape['kind']=='convex':
        world=shape['points']@rotation.T;bottom=world[world[:,2]<=low[2]+.0003].mean(axis=0)
    else:bottom=np.array([0.,0.,low[2]])
    # The COM may initially rise/fall for an irregular support point. Solve
    # v + omega x r = 0, rather than assigning sphere v/r to a fruit or can.
    velocity[2]=-(velocity[:2]@bottom[:2])/bottom[2]
    omega=np.cross(bottom,-velocity)/np.dot(bottom,bottom)
    angular_local=rotation.T@omega
    return dict(rotation=rotation,quat=quat(rotation),velocity=np.r_[velocity,angular_local],
                support=bottom,clearance=-float(low[2]),no_slip_error=float(np.linalg.norm(velocity+np.cross(omega,bottom))))
