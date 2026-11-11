"""Opt-in per-reset DR with exact uniform scaling of the compiled object.

No mesh recompilation and no cumulative scaling. Object mesh vertices, BVH bounds,
visual offsets and analytic collision bounds all derive from nominal snapshots.
"""
import copy
import types
import gymnasium as gym
import mujoco
import numpy as np
from ycb_objects.benchmark_randomization import sample, material
from gym_dcmm.envs.object_test import set_constants_preserving_state

VERSION='physical_dr_episode_v1'

class EpisodeRandomization(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        env.reset()  # Snapshot the fully applied nominal material, not constructor XML defaults.
        leaf=env.unwrapped;self.leaf=leaf
        m=leaf.Dcmm.model;bid=m.body('object').id
        self.bid=bid;self.ids=np.flatnonzero(m.geom_bodyid==bid)
        self.meshes=sorted(set(int(m.geom_dataid[g]) for g in self.ids if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH))
        for mesh in self.meshes:
            assert all(g in self.ids for g in np.flatnonzero((m.geom_type==mujoco.mjtGeom.mjGEOM_MESH)&(m.geom_dataid==mesh))), 'Object mesh shared with robot'
        self.nominal={k:np.array(getattr(m,k)[self.ids],copy=True) for k in ('geom_pos','geom_size','geom_rbound','geom_aabb','geom_friction')}
        self.mass=float(m.body_mass[bid]);self.inertia=m.body_inertia[bid].copy();self.ipos=m.body_ipos[bid].copy()
        self.mesh_snap={}
        for mid in self.meshes:
            v=slice(int(m.mesh_vertadr[mid]),int(m.mesh_vertadr[mid]+m.mesh_vertnum[mid]))
            b=slice(int(m.mesh_bvhadr[mid]),int(m.mesh_bvhadr[mid]+m.mesh_bvhnum[mid]))
            self.mesh_snap[mid]=(v,m.mesh_vert[v].copy(),b,m.bvh_aabb[b].copy(),m.mesh_pos[mid].copy(),m.mesh_scale[mid].copy())
        adr=int(m.body_bvhadr[bid]);num=int(m.body_bvhnum[bid]);self.body_bvh=slice(adr,adr+num) if adr>=0 else slice(0,0)
        self.body_bounds=m.bvh_aabb[self.body_bvh].copy()
        pair='benchmark_surface' if hasattr(leaf,'benchmark_shape') else ('roll_physical_table' if leaf.object_motion=='roll' else 'bounce_v2_ground')
        self.pair=m.pair(pair).id;self.friction=m.pair_friction[self.pair].copy();self.solref=m.pair_solref[self.pair].copy();self.solreffriction=m.pair_solreffriction[self.pair].copy()
        self.shape=copy.deepcopy(getattr(leaf,'benchmark_shape',None))
        self.profile=copy.deepcopy(getattr(leaf,'benchmark_profile',None))
        if self.shape is not None:
            def world_bounds(inner):
                from ycb_objects.benchmark_geometry import bounds
                low,high=bounds(inner.benchmark_shape,inner.Dcmm.data.body('object').xmat.reshape(3,3))
                pos=inner.Dcmm.data.body('object').xpos
                return pos+low,pos+high
            leaf._benchmark_world_bounds=types.MethodType(world_bounds,leaf)
        self.original_reset_simulation=leaf._reset_simulation
        def reset_simulation(inner):
            self.original_reset_simulation()
            self._apply()
        leaf._reset_simulation=types.MethodType(reset_simulation,leaf)
        self.benchmark_randomization=None
        # Also initialize metadata for provenance before the agent's first reset.
        self.reset()

    def _apply(self):
        env=self.leaf;m,d=env.Dcmm.model,env.Dcmm.data
        draw=sample(env.object_motion);draw.update(version=VERSION,sampling='episode_reset')
        s=draw['size_scale'];mass=draw['mass_scale'];self.benchmark_randomization=draw
        for k in ('geom_pos','geom_size','geom_rbound','geom_aabb'):getattr(m,k)[self.ids]=self.nominal[k]*s
        m.geom_friction[self.ids]=self.nominal['geom_friction']
        if self.shape is not None:
            friction_ids=self.ids[self.ids!=env.object_id] if env.object_motion=='bounce' else self.ids
            m.geom_friction[friction_ids,0]*=draw['friction_scale']
            m.geom_friction[friction_ids,2]*=draw['rolling_scale']
        for mid,(v,verts,b,bounds,pos,scale) in self.mesh_snap.items():
            m.mesh_vert[v]=verts*s;m.bvh_aabb[b]=bounds*s;m.mesh_pos[mid]=pos*s;m.mesh_scale[mid]=scale*s
        m.bvh_aabb[self.body_bvh]=self.body_bounds*s
        m.body_mass[self.bid]=self.mass*mass;m.body_inertia[self.bid]=self.inertia*mass*s*s;m.body_ipos[self.bid]=self.ipos*s
        f=self.friction.copy();f[:2]*=draw['friction_scale'];f[3:]*=draw['rolling_scale'];m.pair_friction[self.pair]=f
        m.pair_solref[self.pair]=self.solref;m.pair_solreffriction[self.pair]=self.solreffriction
        if env.object_motion=='bounce':
            m.pair_solref[self.pair,1]*=draw['damping_scale'];m.pair_solreffriction[self.pair,1]*=draw['damping_scale']
        shape=None
        if self.shape is not None:
            shape=copy.deepcopy(self.shape)
            for k in ('radius','marker_radius'):shape[k]*=s
            shape['points']*=s;shape['extents']=(np.asarray(shape['extents'])*s).tolist()
            shape['mass']*=mass;shape['inertia']*=mass*s*s
            env.benchmark_shape=shape;env.benchmark_profile=material(self.profile,draw)
        env.random_mass=float(m.body_mass[self.bid])
        rotation=d.body('object').xmat.reshape(3,3).copy()
        if env.object_motion=='roll':
            from ycb_objects.benchmark_geometry import bounds
            radius=float(m.geom_size[env.object_id,0])
            low=bounds(shape,rotation)[0] if shape is not None else np.full(3,-radius)
            bottom=(shape['points']@rotation.T) if shape is not None and shape['kind']=='convex' else None
            bottom=bottom[bottom[:,2]<=low[2]+.0003].mean(axis=0) if bottom is not None else np.array([0.,0.,low[2]])
            table=m.geom('table_surface').id;top=float(d.geom_xpos[table,2]+m.geom_size[table,2])
            env.object_pos3d[2]=top-low[2]+(.0003 if shape is not None else .0002)
            velocity=env.object_vel6d[:3].copy();velocity[2]=-(velocity[:2]@bottom[:2])/bottom[2]
            omega=np.cross(bottom,-velocity)/np.dot(bottom,bottom)
            env.object_vel6d=np.r_[velocity,rotation.T@omega]
        elif self.shape is None:
            # Preserve the legacy launch's sampled dimensionless spin factor.
            env.object_vel6d[3:5]/=s
        else:
            env.object_pos3d[2]+=shape['radius']-self.shape['radius']
        set_constants_preserving_state(m,d)
        env.Dcmm.set_throw_pos_vel(pose=np.r_[env.object_pos3d,env.object_q],velocity=env.object_vel6d)
        mujoco.mj_forward(m,d)
        viewer=getattr(env.Dcmm,'viewer',None)
        if viewer is not None:
            for mid in self.meshes:viewer.update_mesh(mid)
        renderer=getattr(env,'mujoco_renderer',None)
        if renderer is not None:
            for context in renderer._viewers.values():
                context.make_context_current()
                for mid in self.meshes:mujoco.mjr_uploadMesh(m,context.con,mid)
        if shape is not None:
            e=env.benchmark_episode
            e.update(mass_kg=shape['mass'],dimensions_m=shape['extents'],inertia=shape['inertia'].tolist(),material=env.benchmark_profile,randomization=draw)
            if env.object_motion=='roll':
                e['launch'].update(support_point=bottom.tolist(),initial_support_clearance=.0003,
                    no_slip_error=float(np.linalg.norm(velocity+np.cross(omega,bottom))))
            if env.object_motion=='bounce':env.bounce_v2_episode=e
        else:
            e=env.roll_physics_episode if env.object_motion=='roll' else env.bounce_v2_episode
            e.update(**({'mass_kg':env.random_mass,'radius_m':float(m.geom_size[env.object_id,0])} if env.object_motion=='roll' else {'mass':env.random_mass,'radius':float(m.geom_size[env.object_id,0])}))
            e['sliding_friction']=float(f[0])
            if env.object_motion=='roll':e['rolling_length_m']=float(f[3])
            else:e['damping_ratio']=float(m.pair_solref[self.pair,1])

    def _annotate(self, info):
        info['object_benchmark']=dict(info['object_benchmark'],randomization=copy.deepcopy(self.benchmark_randomization))
        info['benchmark_randomization_version']=VERSION

    def reset(self,**kwargs):
        obs,info=self.env.reset(**kwargs);self._annotate(info);return obs,info

    def step(self,action):
        obs,reward,terminated,truncated,info=self.env.step(action);self._annotate(info)
        return obs,reward,terminated,truncated,info
