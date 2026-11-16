"""Opt-in palm coordinates for Roll Catch; legacy metrics remain separate."""
import numpy as np

VERSION='palm_z_v1'

def enabled(env):
    return getattr(env,'roll_palm_version','legacy')==VERSION

def support_distance(env,normal):
    shape=getattr(env,'benchmark_shape',None)
    if shape is not None and shape['kind']!='sphere':
        points=np.asarray(shape['points'])
        rotation=env.Dcmm.data.body('object').xmat.reshape(3,3)
        # Distance from COM to the side touching the supporting palm.
        return max(.001,float(np.max(-(points@rotation.T)@normal)))
    return float(env.Dcmm.model.geom_size[env.object_id,0])

def capture_point(env):
    data=env.Dcmm.data
    rotation=data.body('link6').xmat.reshape(3,3)
    normal=rotation[:,2]
    if not hasattr(env,'_roll_palm_surface_local'):
        from ycb_objects.benchmark_geometry import mesh_points
        local=(mesh_points(env.Dcmm.model,data,env.hand_start_id)-data.body('link6').xpos)@rotation
        # Robust mesh-derived palm-face estimate. This is a geometric prior,
        # not a measured real-hand contact plane.
        center=np.mean(local,axis=0)
        center[2]=np.percentile(local[:,2],90)
        env._roll_palm_surface_local=center
    surface=data.body('link6').xpos+rotation@env._roll_palm_surface_local
    return surface+normal*support_distance(env,normal)

def scoop_weight(env):
    # Preserve pre-contact guidance; release orientation after acquiring contact.
    state=getattr(env,'roll_grip_state',{})
    first=state.get('first_contact')
    if first is None:return 1.
    return float(np.clip(1.-(env.Dcmm.data.time-first)/.3,0.,1.))
