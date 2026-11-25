"""Stage two curriculum: physical carry and throw, with no scripted actuation."""
import copy
import numpy as np
from gymnasium import spaces
from basket_autonomous.environment import AutonomousTeacherEnv as Base

# Each row: target y range, carry/fixed/mobile sampling probabilities.
LEVELS=(([1.35,1.5],[.7,.3,0.]),([1.6,1.9],[.7,.3,0.]),
        ([1.6,1.9],[.1,.2,.7]),([1.8,2.4],[.1,.2,.7]))

class AutonomousTeacherEnv(Base):
    def __init__(self,options=None,**kwargs):
        self.curriculum_options=copy.deepcopy(options or {})
        self.curriculum_task='throw';self._carry_stable_seconds=0.
        super().__init__(options,**kwargs)
        self.observation_space=spaces.Box(-np.inf,np.inf,(66,),dtype=np.float32)
    def reset(self,seed=None,options=None):
        if seed is not None:self.np_random=np.random.default_rng(seed)
        level=int(self.curriculum_options.get('curriculum_level',0))
        yrange,probabilities=LEVELS[level]
        forced=self.curriculum_options.get('evaluation_task')
        chosen=forced or str(self.np_random.choice(['carry','fixed','mobile'],p=probabilities))
        self.curriculum_task='carry' if chosen=='carry' else 'throw'
        self.options['fixed_replay_probability']=float(chosen=='fixed')
        yrange=([1.8,2.4] if forced=='far_mobile' else yrange)
        self.basket_cfg.mobile_target_y_range=yrange
        self.basket_cfg.mobile_target_x_range=[-.1,.1] if level==0 and forced!='far_mobile' else [-.25,.25]
        self._carry_stable_seconds=0.
        obs,info=super().reset(seed=seed,options=options)
        info['curriculum_task']=chosen
        return obs,info
    def observation(self):
        return np.r_[super().observation(),float(self.curriculum_task=='carry')].astype(np.float32)
    def step(self,action):
        obs,reward,terminated,truncated,info=super().step(action)
        info['curriculum_task']='carry' if self.curriculum_task=='carry' else ('fixed' if self.fixed_episode else 'mobile')
        info['carry_success']=False
        if self.curriculum_task=='carry':
            diagnostics=info['carry_diagnostics'];distance=self._navigation_distance()
            stable=(info['hand_contact'] and not self._released_once and distance<=.05
                and info['base_speed']<.05 and diagnostics.get('slip_speed',1.)<.25
                and diagnostics.get('in_support_envelope',False))
            self._carry_stable_seconds=self._carry_stable_seconds+.04 if stable else 0.
            success=self._carry_stable_seconds>=.3 and not self.basket_robot_collision and not self.basket_arm_self_collision
            if self._released_once:
                terminated=True;self.terminated=True;self.terminated_reason='carry_release'
            if success:
                terminated=True;truncated=False;self.terminated=True;self.terminated_reason='carry_success'
            # Replace throw-specific terms; physical dynamics and all action channels remain untouched.
            terms={k:v for k,v in info['reward_terms'].items() if k not in ('score','failure','release','release_distance','ballistic_progress')}
            terms['carry_success']=30.*float(success)
            terms['failure']=-20.*float((terminated or truncated) and not success)
            reward=sum(terms.values())
            # Parent totals contain throw rewards; keep a separate authoritative carry total.
            if not hasattr(self,'_carry_totals') or self.steps==1:self._carry_totals={}
            for k,v in terms.items():self._carry_totals[k]=self._carry_totals.get(k,0.)+v
            info.update(success=success,carry_success=success,safe_basket_success=False,
                terminated_reason=self.terminated_reason,reward_terms=terms,reward_totals=self._carry_totals.copy())
        info['task_success']=bool(info['carry_success'] if self.curriculum_task=='carry' else info['safe_basket_success'])
        return obs,float(reward),terminated,truncated,info
