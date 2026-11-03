"""Opt-in new-object motion benchmark. Existing three-ball routes are untouched."""
import hashlib
import inspect
import json
import math
from pathlib import Path
import random
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from ycb_objects.benchmark_geometry import geometry,inject,bounds,roll_state
from ycb_objects.loader import remap_contacts
from gym_dcmm.utils.roll_speed import speed_range

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_CATALOG=ROOT/'ycb_objects/calibration_data/object_benchmark_v1.json'


def catalog(path=None):
    path=Path(path or DEFAULT_CATALOG).resolve();raw=path.read_bytes();data=json.loads(raw)
    if data.get('schema')!='object_motion_benchmark_v1':raise ValueError('Invalid benchmark catalog schema')
    allowed={'roll':{'orange','apple','lemon','tomato_soup_can'},'bounce':{'racquetball','golf_ball','softball'}}
    for motion,names in allowed.items():
        if not data.get(motion) or set(data[motion])-names:raise ValueError('Invalid benchmark objects: '+motion)
        for name,p in data[motion].items():
            for key,lo,hi in [('sliding_friction',0,3),('torsional_length_m',0,.005),('rolling_length_m',0,.005)]:
                value=p.get(key,0.)
                if not isinstance(value,(int,float)) or not math.isfinite(value) or not lo<=value<=hi:raise ValueError('Invalid '+name+'.'+key)
            if motion=='bounce':
                for key,lo,hi in [('normal_cor_target',.2,.95),('damping_ratio',.04,1.),('time_constant',.008,.05),('inertia_factor',.1,2/3),('horizontal_retention_prior',.2,1.)]:
                    if not lo<=float(p[key])<=hi:raise ValueError('Invalid '+name+'.'+key)
    return dict(data,path=str(path),sha256=hashlib.sha256(raw).hexdigest())


def choose(motion,test_object,mode,weights,data):
    names=list(data[motion])
    if mode not in ('fixed','mixed'):raise ValueError('Benchmark mode must be fixed or mixed')
    if mode=='mixed':
        if test_object!='default':raise ValueError('Mixed benchmark requires test_object=default')
        probabilities=None
        if weights is not None:
            if not isinstance(weights,dict) or not weights or set(weights)-set(names):raise ValueError('Invalid benchmark subset/weights')
            names=list(weights);values=np.asarray(list(weights.values()),dtype=float)
            if not np.isfinite(values).all() or (values<0).any() or values.sum()<=0:raise ValueError('Invalid benchmark weights')
            probabilities=values/values.sum()
        return str(np.random.choice(names,p=probabilities))
    if weights is not None:raise ValueError('Weights require mixed benchmark')
    name=test_object.removeprefix('ycb:')
    if name not in names:raise ValueError('Choose a benchmark '+motion+' object from '+str(names))
    return name


def bounce_launch(profile,radius,rng=np.random):
    """Passive ballistic launch; modify only the initial state, never apply forces.

    COR sets required incoming energy. A lower-COR softball requires a stronger
    downward launch and uses a lower apex range; material restitution is not raised.
    Horizontal retention is an explicit prior, verified by the passive motion probe.
    """
    e=profile['normal_cor_target'];apex=float(rng.uniform(*profile['rebound_apex_m']))
    energy_height=apex/e**2
    height=min(float(rng.uniform(*profile['launch_height_m'])),energy_height-.05)
    downward=math.sqrt(max(0,2*9.81*(energy_height-height)))
    if height<.5 or downward>8.5:raise ValueError('Requested rebound outside supported launch envelope')
    hit=(math.sqrt(downward**2+2*9.81*height)-downward)/9.81
    up=math.sqrt(2*apex/9.81)
    y=float(rng.uniform(1.5,1.9));apex_y=float(rng.uniform(.55,.85))
    speed=(y-apex_y)/(hit+profile['horizontal_retention_prior']*up)
    if speed>3.5:raise ValueError('Required horizontal launch exceeds 3.5m/s')
    heading=math.radians(float(rng.uniform(-8,8)))
    velocity=np.array([speed*math.sin(heading),-speed*math.cos(heading),-downward,0.,0.,0.])
    return np.array([rng.uniform(-.15,.15),y,height+radius]),velocity,dict(
        requested_apex_clearance_m=apex,nominal_cor=e,predicted_apex_y=apex_y,
        downward_speed=downward,horizontal_speed=speed,predicted_first_impact_seconds=hit,
        predicted_apex_seconds=hit+up,horizontal_retention_prior=profile['horizontal_retention_prior'])


def make_object_benchmark(object_benchmark='off',benchmark_catalog=None,benchmark_options=None,
                          test_object='default',**kwargs):
    motion=kwargs.get('object_motion');kind=str(object_benchmark)
    if kind!=motion+'_v1' or motion not in ('roll','bounce') or kwargs.get('object_eval',False):raise ValueError('Invalid benchmark motion/object_eval')
    if kwargs.get('roll_wait_guard',False) or kwargs.get('roll_reward_version','legacy')!='legacy':raise ValueError('Benchmark uses legacy Roll reward with no hard guard')
    opts=dict(benchmark_options or {})
    if set(opts)-{'heading_degrees','distance_m'}:raise ValueError('Unknown benchmark options')
    data=catalog(benchmark_catalog)
    mode=kwargs.get(motion+'_ball_mode','fixed');weights=kwargs.get(motion+'_ball_weights')
    selected=choose(motion,test_object,mode,weights,data);profile=data[motion][selected]
    shape=geometry(selected,profile['kind'],profile.get('inertia_factor',.4))
    if motion=='roll':
        if kwargs.get('task')=='Tracking':
            from gym_dcmm.envs.DcmmVecEnv import RollTrackingEnv as Base
        else:
            from gym_dcmm.envs.DcmmVecEnv_roll_645edc4 import DcmmVecEnv as Base
        speed=speed_range(kwargs.get('roll_speed')) or (.8,1.3)
    elif kwargs.get('task')=='Tracking':
        from gym_dcmm.envs.DcmmVecEnv_bounce_july22 import DcmmVecEnv as Base
    elif kwargs.get('bounce_grasp_version','legacy')=='controlled_hold_v1':
        from gym_dcmm.envs.bounce_grasp import BounceGraspEnv as Base
    else:
        from gym_dcmm.envs.DcmmVecEnv import BounceEnv as Base
    accepted=inspect.signature(Base.__init__).parameters
    # BounceGraspEnv forwards historical arguments, just like physical_v2.
    if motion=='bounce' and kwargs.get('bounce_grasp_version')=='controlled_hold_v1':
        from gym_dcmm.envs.DcmmVecEnv import BounceEnv
        accepted=inspect.signature(BounceEnv.__init__).parameters

    class BenchmarkObject(Base):
        def _reset_object_roll(self):
            return inject(super()._reset_object_roll(),shape,profile,'table_surface')

        def _reset_object_bounce(self):
            return inject(super()._reset_object_bounce(),shape,profile,'floor')

        def _reset_simulation(self):
            super()._reset_simulation()
            model,d=self.Dcmm.model,self.Dcmm.data
            if motion=='roll':
                heading_range=opts.get('heading_degrees',[-10,10]);distance=opts.get('distance_m',[2.0,2.2])
                initial=roll_state(shape,float(np.random.uniform(*speed)),math.radians(np.random.uniform(*heading_range)),np.random.uniform(-math.pi,math.pi))
                table=model.geom('table_surface').id
                top=float(d.geom_xpos[table,2]+model.geom_size[table,2])
                self.object_pos3d=np.array([np.random.uniform(-.15,.15),np.random.uniform(*distance),top+initial['clearance']+.0003])
                self.object_q=initial['quat'];self.object_vel6d=initial['velocity']
                description=dict(no_slip_error=initial['no_slip_error'],support_point=initial['support'].tolist(),
                                 axis_world=initial['rotation'][:,2].tolist(),initial_support_clearance=.0003)
            else:
                from gym_dcmm.envs.bounce_physical_v2 import DT,SUBSTEPS
                model.opt.timestep=DT;model.opt.integrator=mujoco.mjtIntegrator.mjINT_RK4
                self._bounce_physics_substeps=SUBSTEPS;self.fps=1/(self.steps_per_policy*DT*SUBSTEPS)
                self.object_pos3d,self.object_vel6d,description=bounce_launch(profile,shape['radius'])
                self.object_q=np.array([1.,0,0,0])
            self.random_mass=shape['mass'];self.object_static_time=0.
            self.Dcmm.set_throw_pos_vel(pose=np.r_[self.object_pos3d,self.object_q],velocity=self.object_vel6d)
            mujoco.mj_forward(model,d)
            self.benchmark_episode=dict(version=kind,object=selected,shape=shape['kind'],mass_kg=shape['mass'],
                dimensions_m=shape['extents'],inertia=shape['inertia'].tolist(),inertia_basis=shape['inertia_basis'],
                surface_id=data['surface_id'],material=profile,launch=description,
                sampling='fixed_object_per_environment',status=data['status'])
            # Keep compatibility with the current Bounce hold episode logger.
            if motion=='bounce':self.bounce_v2_episode=self.benchmark_episode

        def _benchmark_world_bounds(self):
            lo,hi=bounds(shape,self.Dcmm.data.body('object').xmat.reshape(3,3))
            p=self.Dcmm.data.body('object').xpos
            return p+lo,p+hi

        def _get_contacts(self):
            result=super()._get_contacts()
            model=self.Dcmm.model;ids=np.flatnonzero(model.geom_bodyid==model.body('object').id)
            return remap_contacts(result,np.asarray(self.Dcmm.data.contact.geom).reshape(-1,2),ids,self.object_id)

        def reset(self,*,seed=None,options=None):
            if options:raise ValueError('Benchmark reset options unsupported')
            if seed is not None:np.random.seed(seed);random.seed(seed)
            obs,info=super().reset()
            info.update(object_benchmark=self.benchmark_episode,benchmark_initial_position=self.object_pos3d.copy(),benchmark_initial_velocity=self.object_vel6d.copy())
            return obs,info

        def step(self,action):
            obs,reward,terminated,truncated,info=super().step(action)
            info['object_benchmark']=self.benchmark_episode
            return obs,reward,terminated,truncated,info

    parameters={k:v for k,v in kwargs.items() if k in accepted}
    if motion=='bounce' and kwargs.get('bounce_grasp_version')=='controlled_hold_v1':parameters['bounce_grasp']=kwargs.get('bounce_grasp')
    env=BenchmarkObject(**parameters)
    env.roll_log=bool(kwargs.get('roll_log',False))
    env.benchmark_shape=shape;env.benchmark_profile=profile;env.benchmark_catalog=data
    return env


def write_provenance(output,config):
    data=catalog(config.get('benchmark_catalog'))
    names=['train_DCMM.py','configs/config.yaml','gym_dcmm/envs/object_benchmark.py','ycb_objects/benchmark_geometry.py','gym_dcmm/utils/roll_evaluation.py']
    result=dict(version=config['object_benchmark'],catalog=data,options=config.get('benchmark_options'),
                sampling='one_object_sampled_at_environment_creation_no_mesh_recompile_on_reset',
                reward='existing task reward; shape-aware departure for non-spheres',
                sources={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names})
    (Path(output)/'object_benchmark_provenance.json').write_text(json.dumps(result,indent=2))
