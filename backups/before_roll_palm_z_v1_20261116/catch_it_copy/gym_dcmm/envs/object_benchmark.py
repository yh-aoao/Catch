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
import gymnasium as gym
from ycb_objects.benchmark_geometry import geometry,inject,bounds,roll_state
from ycb_objects.loader import remap_contacts
from gym_dcmm.utils.roll_speed import speed_range

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_CATALOG=ROOT/'ycb_objects/calibration_data/object_benchmark_v1.json'
LEGACY_OBJECTS={'original','tennis_ball','baseball'}


def catalog(path=None):
    path=Path(path or DEFAULT_CATALOG).resolve();raw=path.read_bytes();data=json.loads(raw)
    if data.get('schema')!='object_motion_benchmark_v1':raise ValueError('Invalid benchmark catalog schema')
    allowed={'roll':{'orange','apple','lemon','tomato_soup_can'}|LEGACY_OBJECTS,
             'bounce':{'racquetball','golf_ball','softball'}|LEGACY_OBJECTS}
    for motion,names in allowed.items():
        if not data.get(motion) or set(data[motion])-names:raise ValueError('Invalid benchmark objects: '+motion)
        for name,p in data[motion].items():
            if name in LEGACY_OBJECTS:
                if p!={'route':'three_ball_baseline'}:
                    raise ValueError('Legacy objects must use the unchanged three_ball_baseline route')
                continue
            for key,lo,hi in [('sliding_friction',0,3),('torsional_length_m',0,.005),('rolling_length_m',0,.005)]:
                value=p.get(key,0.)
                if not isinstance(value,(int,float)) or not math.isfinite(value) or not lo<=value<=hi:raise ValueError('Invalid '+name+'.'+key)
            if motion=='roll' and p.get('speed_mps') is not None:
                speed_range(p['speed_mps'])
            if motion=='bounce':
                for key,lo,hi in [('normal_cor_target',.2,.95),('damping_ratio',.04,1.),('time_constant',.003,.05),('inertia_factor',.1,2/3),('horizontal_retention_prior',.2,1.)]:
                    if not lo<=float(p[key])<=hi:raise ValueError('Invalid '+name+'.'+key)
    return dict(data,path=str(path),sha256=hashlib.sha256(raw).hexdigest())


class LegacyBenchmarkTarget(gym.Wrapper):
    """Attach sampling metadata without replacing old physics/reward/reset logic."""
    def __init__(self,env,name,data):
        super().__init__(env)
        self.benchmark_object=name
        self.benchmark_profile=data[env.object_motion][name]
        self.benchmark_catalog=data
        self.benchmark_episode=dict(object=name,route='three_ball_baseline',sampling='fixed_object_per_environment')

    def _annotate(self,info):
        physics=(getattr(self.env,'roll_physics_episode',None) if self.env.object_motion=='roll'
                 else getattr(self.env,'bounce_v2_episode',None))
        self.benchmark_episode=dict(object=self.benchmark_object,route='three_ball_baseline',
                                    sampling='fixed_object_per_environment',physics=physics)
        if 'benchmark_randomization' in self.__dict__:
            self.benchmark_episode['randomization']=self.benchmark_randomization
        info['object_benchmark']=self.benchmark_episode

    def reset(self,**kwargs):
        obs,info=self.env.reset(**kwargs);self._annotate(info)
        return obs,info

    def step(self,action):
        obs,reward,terminated,truncated,info=self.env.step(action);self._annotate(info)
        return obs,reward,terminated,truncated,info


def legacy_target(motion,name,data,kwargs,draw=None):
    parameters=dict(kwargs)
    parameters.update(test_object='default' if name=='original' else 'ycb:'+name)
    parameters[motion+'_ball_mode']='fixed'
    parameters[motion+'_ball_weights']=None
    if motion=='roll':
        from gym_dcmm.envs.roll_physical import make_roll_physical,DEFAULT_MATERIALS
        parameters['roll_physics']='physical_v1'
        parameters['roll_material_file']=str(DEFAULT_MATERIALS)
        if parameters.get('roll_speed') is None:
            parameters['roll_speed']=[.8,1.3]
        factory=make_roll_physical
    else:
        from gym_dcmm.envs.bounce_physical_v2 import make_bounce_physical_v2
        parameters['bounce_physics']='physical_v2'
        parameters['bounce_launch']='baseball_intercept_v1'
        parameters['bounce_material_file']=str(ROOT/'ycb_objects/calibration_data/balls.three_ball_training_v1.json')
        factory=make_bounce_physical_v2
    if draw is None:
        env=factory(**parameters)
    else:
        from ycb_objects.benchmark_randomization import legacy_environment
        env=legacy_environment(factory,parameters,parameters[motion+'_material_file'],name,motion,draw)
    wrapped=LegacyBenchmarkTarget(env,name,data)
    if draw is not None:
        wrapped.benchmark_randomization=draw
    return wrapped


def choose(motion,test_object,mode,weights,data,slot=None,num_slots=None):
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
        if slot is not None:
            probabilities=np.full(len(names),1/len(names)) if probabilities is None else probabilities
            index=min(len(names)-1,int(np.searchsorted(np.cumsum(probabilities),(slot+.5)/num_slots)))
            return names[index]
        return str(np.random.choice(names,p=probabilities))
    if weights is not None:raise ValueError('Weights require mixed benchmark')
    name='original' if test_object in ('default','sphere') else test_object.removeprefix('ycb:')
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
                          test_object='default',benchmark_slot=None,benchmark_num_envs=None,
                          benchmark_seed=123,**kwargs):
    motion=kwargs.get('object_motion');kind=str(object_benchmark)
    if motion not in ('roll','bounce') or kind!=motion+'_v1' or kwargs.get('object_eval',False):raise ValueError('Invalid benchmark motion/object_eval')
    if kwargs.get('roll_wait_guard',False) or kwargs.get('roll_reward_version','legacy')!='legacy':raise ValueError('Benchmark uses legacy Roll reward with no hard guard')
    if kwargs.get('ycb_collision','auto')!='auto' or kwargs.get('ycb_joint_damping') is not None:
        raise ValueError('Benchmark owns collision geometry and free-joint damping')
    if kwargs.get(motion+'_material_file'):
        raise ValueError('Use benchmark_catalog; clear the legacy material_file for this benchmark')
    opts=dict(benchmark_options or {})
    if set(opts)-{'heading_degrees','distance_m','domain_randomization'}:raise ValueError('Unknown benchmark options')
    dr=opts.get('domain_randomization','off')
    if dr not in ('off','physical_dr_v1','physical_dr_episode_v1'):raise ValueError('Unknown benchmark domain_randomization')
    if dr=='physical_dr_episode_v1':
        from ycb_objects.episode_randomization import EpisodeRandomization
        base=make_object_benchmark(object_benchmark=object_benchmark,benchmark_catalog=benchmark_catalog,
            benchmark_options=dict(opts,domain_randomization='off'),test_object=test_object,
            benchmark_slot=benchmark_slot,benchmark_num_envs=benchmark_num_envs,benchmark_seed=benchmark_seed,**kwargs)
        return EpisodeRandomization(base)
    for key,default,low,high in [('heading_degrees',[-10,10],-30,30),('distance_m',[2.,2.2],1.1,3.5)]:
        value=np.asarray(opts.get(key,default),dtype=float)
        if value.shape!=(2,) or not np.isfinite(value).all() or not low<=value[0]<=value[1]<=high:raise ValueError('Invalid benchmark '+key)
    if benchmark_slot is not None:
        if not 0<=benchmark_slot<benchmark_num_envs:raise ValueError('Invalid benchmark slot')
        if benchmark_seed>=0:
            np.random.seed(benchmark_seed+benchmark_slot);random.seed(benchmark_seed+benchmark_slot)
    data=catalog(benchmark_catalog)
    mode=kwargs.get(motion+'_ball_mode','fixed');weights=kwargs.get(motion+'_ball_weights')
    selected=choose(motion,test_object,mode,weights,data,benchmark_slot,benchmark_num_envs);profile=data[motion][selected]
    draw=None
    if dr!='off':
        from ycb_objects.benchmark_randomization import sample,material,scaled_geometry
        draw=sample(motion)
    if selected in LEGACY_OBJECTS:
        return legacy_target(motion,selected,data,kwargs,draw)
    shape=geometry(selected,profile['kind'],profile.get('inertia_factor',.4))
    if draw is not None:
        shape=scaled_geometry(shape,draw);profile=material(profile,draw)
    if motion=='roll':
        if kwargs.get('task')=='Tracking':
            from gym_dcmm.envs.DcmmVecEnv import RollTrackingEnv as Base
        else:
            from gym_dcmm.envs.DcmmVecEnv_roll_645edc4 import DcmmVecEnv as Base
        speed=speed_range(kwargs.get('roll_speed')) or speed_range(profile.get('speed_mps')) or (.8,1.3)
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
                heading_range=opts.get('heading_degrees',[-10,10]);distance=opts.get('distance_m',profile.get('distance_m',[2.0,2.2]))
                initial=roll_state(shape,float(np.random.uniform(*speed)),math.radians(np.random.uniform(*heading_range)),np.random.uniform(-math.pi,math.pi))
                table=model.geom('table_surface').id
                top=float(d.geom_xpos[table,2]+model.geom_size[table,2])
                self.object_pos3d=np.array([np.random.uniform(-.15,.15),np.random.uniform(*distance),top+initial['clearance']+.0003])
                self.object_q=initial['quat'];self.object_vel6d=initial['velocity']
                description=dict(no_slip_error=initial['no_slip_error'],support_point=initial['support'].tolist(),
                                 axis_world=initial['rotation'][:,2].tolist(),initial_support_clearance=.0003,
                                 configured_speed_mps=list(speed),configured_distance_m=list(distance))
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
            if draw is not None:self.benchmark_episode['randomization']=draw
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
    env.benchmark_object=selected
    if draw is not None:env.benchmark_randomization=draw
    return env


def make_benchmark_vector(num_envs,benchmark_backend='async',**kwargs):
    """Explicit ranks avoid identical shape choices in forked vector workers."""
    from functools import partial
    from gymnasium.experimental.vector import SyncVectorEnv,AsyncVectorEnv
    kwargs.pop('max_episode_steps',None)
    factories=[partial(make_object_benchmark,benchmark_slot=i,benchmark_num_envs=num_envs,**kwargs) for i in range(num_envs)]
    if benchmark_backend=='sync':return SyncVectorEnv(factories)
    if benchmark_backend=='async':return AsyncVectorEnv(factories)
    raise ValueError('Unknown benchmark vector backend')


def write_provenance(output,config,env=None):
    data=catalog(config.get('benchmark_catalog'))
    names=['train_DCMM.py','configs/config.yaml','gym_dcmm/envs/object_benchmark.py','ycb_objects/benchmark_geometry.py','ycb_objects/benchmark_randomization.py','ycb_objects/episode_randomization.py','gym_dcmm/utils/roll_evaluation.py',
           'gym_dcmm/envs/roll_physical.py','gym_dcmm/envs/bounce_physical_v2.py',
           'ycb_objects/calibration_data/balls.roll_table_priors_v1.json','ycb_objects/calibration_data/balls.three_ball_training_v1.json']
    result=dict(version=config['object_benchmark'],catalog=data,options=config.get('benchmark_options'),
                sampling='one_object_sampled_at_environment_creation_no_mesh_recompile_on_reset',
                reward='existing task reward; shape-aware departure for non-spheres',
                sources={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names})
    if config.get('benchmark_options',{}).get('domain_randomization')=='physical_dr_episode_v1':
        result['physics_sampling']='every_episode_reset; object identity fixed per environment; ranges unchanged from physical_dr_v1'
        result['environment_samples_note']='initial snapshots only; subsequent reset samples are in episode object_benchmark metadata'
    if env is not None and config.get('benchmark_options',{}).get('domain_randomization','off')!='off':
        result['environment_samples']=[dict(slot=i,object=name,randomization=draw)
            for i,(name,draw) in enumerate(zip(env.call('benchmark_object'),env.call('benchmark_randomization')))]
    if config.get('catch_warm_start',False):
        checkpoint=Path(config['checkpoint_catching']).resolve()
        result['catch_warm_start']=dict(source=str(checkpoint),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            initialization='full Catch model and all normalization statistics; fresh optimizer, counters, best score and RNG')
        for n in ['gym_dcmm/algs/ppo_dcmm_645edc4/ppo_dcmm_catch_two_stage.py','gym_dcmm/algs/ppo_dcmm_5fe75d5f/ppo_dcmm_catch_hold.py']:
            result['sources'][n]=hashlib.sha256((ROOT/n).read_bytes()).hexdigest()
    (Path(output)/'object_benchmark_provenance.json').write_text(json.dumps(result,indent=2))
