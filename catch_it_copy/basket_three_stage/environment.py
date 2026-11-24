"""Three-stage curriculum using real Catch pose groups and an observable task cue."""
import copy
import numpy as np
import gymnasium.spaces as spaces
from basket_autonomous.environment import AutonomousTeacherEnv as BaseEnv
from basket_autonomous.carry_reward import LOW,HIGH

class AutonomousTeacherEnv(BaseEnv):
    observation_version='robot36_ball6_goal3_previous20_task1_v1'
    def __init__(self,options=None,**kwargs):
        options=copy.deepcopy(options or {})
        self.curriculum_stage=options.get('curriculum_stage')
        if self.curriculum_stage not in ('hold','throw','mobile'):raise ValueError('Expected hold/throw/mobile stage')
        self._episode_task='hold';self._sampling_condition=('recorded','throw')
        options['training_stage']='mobile' if self.curriculum_stage=='mobile' else 'fixed'
        options['fixed_replay_probability']=0.
        if options.get('arm_velocity_protocol')!='integrated_target_v2' or options.get('release_protocol')!='separated_flight_v2':
            raise ValueError('Three-stage curriculum requires both control/release fixes')
        self._throw_seconds=float(options.get('episode_seconds',10.))
        super().__init__(options,**kwargs)
        self.observation_space=spaces.Box(-np.inf,np.inf,(66,),dtype=np.float32)
        self.groups={task:[p for p in self.pose_bank[1:] if p.get('task')==task] for task in ('throw','roll','bounce')}
        for task,entries in self.groups.items():
            if len(entries)<4:raise ValueError('Need at least four real Catch records for '+task)
            for p in entries:
                if not p.get('contact_validated') or p.get('supported_seconds',0)<.039:
                    raise ValueError('Unvalidated Catch record')
        probs=np.asarray(self.options.get('sampling_probabilities',[.60,.35,.05]),float)
        if probs.shape!=(3,) or not np.isfinite(probs).all() or np.any(probs<0) or not np.isclose(probs.sum(),1.) or probs[:2].sum()<.9:
            raise ValueError('At least 90% reset draws must use recorded Catch states')
        self.sampling_probabilities=probs

    def reset(self,*,seed=None,options=None):
        if seed is not None:self.np_random=np.random.default_rng(seed)
        evaluation=self.options.get('evaluation',False)
        if self.curriculum_stage=='hold':self._episode_task='hold'
        elif self.curriculum_stage=='throw':
            self._episode_task='fixed_throw' if evaluation or self.np_random.random()>=.20 else 'hold'
        else:
            self._episode_task='mobile_throw' if evaluation else self.np_random.choice(
                ['hold','fixed_throw','mobile_throw'],p=[.10,.20,.70])
        mode=self.np_random.choice(['recorded','near','generic'],p=self.sampling_probabilities)
        source=self.np_random.choice(['throw','roll','bounce']) if mode!='generic' else 'generic'
        self._sampling_condition=(mode,source)
        self.training_stage='mobile' if self._episode_task=='mobile_throw' else 'fixed'
        self.options['episode_seconds']=float(self.options.get('hold_max_seconds',4.)) if self._episode_task=='hold' else self._throw_seconds
        self.env_time=self.options['episode_seconds']
        self._stable_hold_seconds=0.
        self._required_hold=float(self.np_random.uniform(.8,1.6))
        obs,info=super().reset(seed=seed,options=options)
        info.update(episode_task=self._episode_task,reset_source_task=source,reset_mode=mode)
        self.reset_stats.update(episode_task=self._episode_task,source_task=source,sampling_mode=mode)
        info['reset_sampling']=copy.deepcopy(self.reset_stats)
        return obs,info

    def _sample_hold_anchor(self):
        mode,task=self._sampling_condition
        if mode=='generic':return self.pose_bank[0]
        group=self.groups[task]
        return group[int(self.np_random.integers(len(group)))]

    def _hold_noise_scales(self,anchor):
        mode,_=self._sampling_condition
        if mode=='recorded':return np.zeros(6),0.,np.zeros(3)
        return (np.asarray(self.options.get('near_arm_jitter_rad',[.035]*6)),
            float(self.options.get('near_hand_jitter_rad',.04)),
            np.asarray(self.options.get('near_ball_jitter_m',[.003,.003,.002])))

    def _accept_hold_placement(self,anchor,placement):
        # Never silently rescue an invalid recorded pose by moving the ball far away.
        return self._sampling_condition[0]=='generic' or placement['lift']<=.050

    def _accept_prepared_hold(self,anchor):
        if self._sampling_condition[0]=='generic':return True
        d=self.Dcmm.data;local=self._carry_measurements()[0]
        return (np.max(np.abs(d.qpos[15:21]-np.asarray(anchor['arm'])))<.15 and
            np.max(np.abs(d.qpos[21:37]-np.asarray(anchor['hand'])))<.60 and
            np.sqrt(np.mean((d.qpos[21:37]-np.asarray(anchor['hand']))**2))<.25 and
            np.linalg.norm(local-np.asarray(anchor['local']))<.025)

    def observation(self):
        return np.r_[super().observation(),float(self._episode_task!='hold')].astype(np.float32)

    def step(self,action):
        previous=self.last_action.copy()
        obs,reward,done,trunc,info=super().step(action)
        hold_success=False
        if self._episode_task=='hold':
            local,relative,_,_=self._carry_measurements();speed=float(np.linalg.norm(relative))
            inside=bool(np.all(local>=LOW) and np.all(local<=HIGH));touch=self.touching()
            stable=touch and inside and speed<.12 and not self._released_once
            dt=self.steps_per_policy*self.Dcmm.model.opt.timestep
            self._stable_hold_seconds=self._stable_hold_seconds+dt if stable else 0.
            hold_success=bool(self._stable_hold_seconds>=self._required_hold and not done)
            terms=dict(hold_stable=.5*dt*float(touch and inside)*np.exp(-(speed/.15)**2),
                hold_slip=-.5*dt*float(touch)*min((speed/.3)**2,1.),time=-.02*dt,
                action_cost=-.002*float(np.mean(self.last_action**2)),
                action_smoothness=-.01*float(np.mean((self.last_action-previous)**2)),
                hold_success=20.*float(hold_success),failure=-15.*float((done or trunc) and not hold_success))
            reward=float(sum(terms.values()))
            for key,value in terms.items():self._hold_totals[key]=self._hold_totals.get(key,0.)+float(value)
            if hold_success:
                self.terminated=True;self.terminated_reason='stable_hold';done=True;trunc=False
            info.update(success=hold_success,safe_basket_success=False,reward_version='three_stage_hold_v1',terminated_reason=self.terminated_reason,
                reward_terms=terms,reward_totals=self._hold_totals.copy())
        if self._episode_task!='hold':info['success']=bool(info.get('safe_basket_success',False))
        info.update(episode_task=self._episode_task,hold_success=hold_success,
            stable_hold_seconds=self._stable_hold_seconds,required_hold_seconds=self._required_hold,
            reset_source_task=self._sampling_condition[1],reset_mode=self._sampling_condition[0],
            reset_sampling=copy.deepcopy(self.reset_stats))
        return obs,reward,done,trunc,info

    def _reset_targets(self):
        super()._reset_targets()
        self._hold_totals={}
