"""Prepare+transport first, then throw from real preparation states. No runtime assistance."""
import copy,pickle
from pathlib import Path
import mujoco
import numpy as np
from gymnasium import spaces
from .physics import AutonomousTeacherEnv as Base
from gym_dcmm.utils.basket_grip import palm_coordinates
from .state import capture,restore,VERSION
LEVELS=([1.35,1.5],[1.6,1.9],[1.8,2.4])

def preparation_metrics(env,info):
    d,m=env.Dcmm.data,env.Dcmm.model
    local=palm_coordinates(d,d.qpos[37:40]);lo=np.array([-.045,-.015,.045]);hi=np.array([.045,.095,.12])
    edge=float(np.linalg.norm(local-np.clip(local,lo,hi)))
    palm_up=float(d.body('link6').xmat.reshape(3,3)[2,2])
    ids=[int(np.flatnonzero(m.jnt_qposadr==i)[0]) for i in range(15,21)]
    margin=min(min(d.qpos[15+i]-m.jnt_range[j,0],m.jnt_range[j,1]-d.qpos[15+i])/
        max(float(np.ptp(m.jnt_range[j])),1e-6) for i,j in enumerate(ids) if m.jnt_limited[j])
    slip=info['carry_diagnostics'].get('slip_speed',1.)
    arm_speed=float(np.linalg.norm(d.qvel[14:20]))
    distance=float(np.linalg.norm(env.basket_center[:2]-d.body('arm_base').xpos[:2]))
    posture_quality=float(np.exp(-(edge/.035)**2)*np.clip((palm_up+.1)/.35,0.,1.)*
        np.clip(margin/.01,0.,1.)*np.exp(-(slip/.18)**2))
    posture_ok=(edge<=1e-6 and palm_up>=.25 and margin>=.01 and slip<.12 and arm_speed<1.)
    ready=(posture_ok and info['hand_contact'] and not env._released_once and .85<=distance<=1.15
        and env._navigation_distance()<=.05)
    return dict(ready=bool(ready),posture_quality=posture_quality,ball_local=local.tolist(),palm_up=palm_up,
        arm_limit_margin=float(margin),arm_speed=arm_speed,slip_speed=slip,base_target_distance=distance)

class AutonomousTeacherEnv(Base):
    def __init__(self,options=None,**kwargs):
        self.settings=copy.deepcopy(options or {});self.training_phase=self.settings.get('two_stage','prepare')
        if self.training_phase not in ('prepare','throw'):raise ValueError('Expected prepare or throw stage')
        self.episode_kind='prepare';self.phase='prepare';self.ready_seconds=0.;self._totals={}
        self.bank=[];self.last_ready_snapshot=None;self._best_posture=0.
        forced=self.settings.get('evaluation_task')
        if self.training_phase=='throw' and forced not in ('full','prepare'):
            path=self.settings.get('ready_bank')
            if not path or not Path(path).is_file():raise ValueError('Stage two needs --ready-bank collected from stage one')
            bank=pickle.loads(Path(path).read_bytes())
            if bank.get('version')!=VERSION or not bank.get('states'):raise ValueError('Invalid ready bank')
            for key in ('pose_bank_entries','arm_jitter_rad','hand_jitter_rad','arm_velocity_protocol','release_protocol','hoop_diameter','release_requires_stop'):
                if bank['source_options'].get(key)!=self.settings.get(key):raise ValueError('Ready bank source differs: '+key)
            self.bank=bank['states']
        super().__init__(options,**kwargs)
        self.observation_space=spaces.Box(-np.inf,np.inf,(66,),dtype=np.float32)
    def observation(self):
        obs=super().observation()
        # Measured relative palm velocity is reproducible across state replay.
        d,m=self.Dcmm.data,self.Dcmm.model;base=d.body('arm_base');palm=d.body('link6')
        a=np.zeros(6);b=np.zeros(6)
        mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,m.body('link6').id,a,0)
        mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,m.body('arm_base').id,b,0)
        obs[9:12]=base.xmat.reshape(3,3).T@(a[3:]-b[3:]-np.cross(b[:3],palm.xpos-base.xpos))
        return np.r_[obs,float(self.phase=='prepare')].astype(np.float32)
    def reset(self,seed=None,options=None):
        rng=np.random.default_rng(seed)
        forced=self.settings.get('evaluation_task')
        self.episode_kind=forced or ('prepare' if self.training_phase=='prepare' else str(rng.choice(['short_throw','full','prepare'],p=[.6,.25,.15])))
        self.phase='throw' if self.episode_kind=='short_throw' else 'prepare'
        level=int(self.settings.get('preparation_level',0));self.basket_cfg.mobile_target_y_range=LEVELS[level]
        self.basket_cfg.mobile_target_x_range=[-.1,.1] if level==0 else [-.25,.25]
        self.options['fixed_replay_probability']=0.
        self.ready_seconds=0.;self._totals={};self._best_posture=0.;self.last_ready_snapshot=None
        obs,info=super().reset(seed=seed,options=options)
        if self.episode_kind=='short_throw':
            restore(self,self.bank[int(self.np_random.integers(len(self.bank)))])
            self.reset_stats=dict(self.reset_stats,replayed_ready_state=True)
            obs=self.observation();info=dict(reset_sampling=copy.deepcopy(self.reset_stats),fixed_episode=False)
        info['episode_kind']=self.episode_kind
        return obs,info
    def step(self,action):
        was_prepare=self.phase=='prepare'
        obs,reward,terminated,truncated,info=super().step(action)
        metrics=preparation_metrics(self,info)
        prepare_success=False
        if was_prepare:
            terms={k:v for k,v in info['reward_terms'].items() if k not in ('score','failure','release','release_distance','ballistic_progress')}
            quality=metrics['posture_quality'] if info['hand_contact'] and not self._released_once else 0.
            terms['prepare_quality']=.3*.04*quality
            terms['prepare_progress']=3.*max(quality-self._best_posture,0.)
            self._best_posture=max(self._best_posture,quality)
            self.ready_seconds=self.ready_seconds+.04 if metrics['ready'] else 0.
            prepare_success=(self.ready_seconds>=.3 and not (terminated or truncated)
                and not self.basket_robot_collision and not self.basket_arm_self_collision)
            if self._released_once:
                terminated=True;truncated=False;self.terminated=True;self.terminated_reason='prepare_release'
            if prepare_success:
                self.last_ready_snapshot=capture(self)
                if self.training_phase=='throw':
                    # Refresh this worker's replay distribution using on-policy arrivals.
                    self.bank.append(self.last_ready_snapshot)
                    if len(self.bank)>1000:self.bank.pop(0)
                if self.episode_kind=='prepare':
                    terminated=True;self.terminated=True;self.terminated_reason='prepare_success'
                else:self.phase='throw';obs[-1]=0.
            terms['prepare_success']=(40. if self.episode_kind=='prepare' else 5.)*float(prepare_success)
            terms['failure']=-20.*float((terminated or truncated) and not prepare_success)
            reward=sum(terms.values());info['reward_terms']=terms
            info.update(success=bool(prepare_success and self.episode_kind=='prepare'),safe_basket_success=False)
        for k,v in info['reward_terms'].items():self._totals[k]=self._totals.get(k,0.)+v
        info.update(episode_kind=self.episode_kind,phase=self.phase,preparation=metrics,prepare_success=bool(prepare_success),
            reached_throw_phase=self.phase=='throw',terminated_reason=self.terminated_reason,
            task_success=bool(prepare_success if self.episode_kind=='prepare' else info['safe_basket_success']),
            reward_totals=self._totals.copy())
        return obs,float(reward),terminated,truncated,info
