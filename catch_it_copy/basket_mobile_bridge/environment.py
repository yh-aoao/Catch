"""Physical carry-to-throw handoff within a single episode; no state resets at handoff."""
import numpy as np
from basket_mobile_curriculum.environment import AutonomousTeacherEnv as Curriculum

def ready_to_throw(info,distance,region_distance):
    diagnostics=info.get('carry_diagnostics',{})
    return (.85<=distance<=1.15 and region_distance<=.05 and info['hand_contact']
        and info['base_speed']<.05 and diagnostics.get('slip_speed',1.)<.25
        and diagnostics.get('in_support_envelope',False))

class AutonomousTeacherEnv(Curriculum):
    def __init__(self,options=None,**kwargs):
        self._throw_ready=False;self._handoff_stable=0.;self._handoff_distance=None
        self._bridge_totals={}
        super().__init__(options,**kwargs)
    def reset(self,seed=None,options=None):
        self._throw_ready=False;self._handoff_stable=0.;self._handoff_distance=None
        self._bridge_totals={}
        return super().reset(seed=seed,options=options)
    def observation(self):
        obs=super().observation()
        if self.curriculum_task!='carry' and not self.fixed_episode:
            obs[-1]=float(not self._throw_ready)
        return obs
    def step(self,action):
        obs,reward,terminated,truncated,info=super().step(action)
        if info['curriculum_task']=='mobile':
            terms=info['reward_terms'].copy();terms['handoff']=0.
            if not self._throw_ready:
                # No throwing credit before a supported, stable arrival; no actuator is overridden.
                for key in ('release','ballistic_progress','score'):
                    terms[key]=min(0.,terms.get(key,0.))
                if self._released_once:
                    terminated=True;truncated=False;self.terminated=True
                    self.terminated_reason='premature_bridge_release'
                    terms['failure']=-20.
                    info.update(success=False,safe_basket_success=False,task_success=False,
                        terminated_reason=self.terminated_reason)
                elif not (terminated or truncated):
                    d=self.Dcmm.data
                    distance=float(np.linalg.norm(self.basket_center[:2]-d.body('arm_base').xpos[:2]))
                    ready=ready_to_throw(info,distance,self._navigation_distance())
                    self._handoff_stable=self._handoff_stable+.04 if ready else 0.
                    if self._handoff_stable>=.12:
                        self._throw_ready=True;self._handoff_distance=distance;terms['handoff']=5.
            for k,v in terms.items():self._bridge_totals[k]=self._bridge_totals.get(k,0.)+v
            reward=sum(terms.values());info.update(reward_terms=terms,reward_totals=self._bridge_totals.copy())
            obs=self.observation()
        info.update(bridge_phase=('throw' if self._throw_ready else 'approach') if info['curriculum_task']=='mobile' else info['curriculum_task'],
            reached_throw_phase=self._throw_ready,handoff_distance=self._handoff_distance)
        return obs,float(reward),terminated,truncated,info
