"""Isolated evaluation instrumentation; production files are never edited."""
import os,sys,runpy
from pathlib import Path
ROOT=Path('/home/isee/Catch/catch_it_copy');sys.path.insert(0,str(ROOT))
from ycb_objects import benchmark_randomization as br
scale=float(os.environ['PROBE_DR_SCALE']);sample_original=br.sample
def sample(*a,**kw):
 d=sample_original(*a,**kw)
 for k in ('size_scale','mass_scale','friction_scale','rolling_scale','damping_scale'):d[k]=1+(d[k]-1)*scale
 return d
br.sample=sample
from gym_dcmm.envs.bounce_grasp import BounceGraspEnv
orig_reset=BounceGraspEnv.reset;orig_state=BounceGraspEnv._grasp_state;orig_step=BounceGraspEnv.step
def reset(self,*a,**kw):
 self._probe=None
 return orig_reset(self,*a,**kw)
def state(self):
 s=orig_state(self)
 if getattr(self,'_grasp_sampling',False):
  t=float(self.Dcmm.data.time)
  if self._probe is None:self._probe=dict(last_time=-1.,ticks=0,contact_ticks=0,first_contact=None,near_ticks=0,near_speed_sum=0.,max_groups=0,gates={},contact_gates={},min_distance=999.)
  p=self._probe
  if t>p['last_time']:
   p['last_time']=t;p['ticks']+=1;p['contact_ticks']+=int(s['hand_contact']);p['min_distance']=min(p['min_distance'],s['distance']);p['max_groups']=max(p['max_groups'],s['contact_groups'])
   gates=dict(capture=s['capture'],groups=s['contact_groups']>=self.grasp_cfg.min_contact_groups,span=s['contact_span']>=self.grasp_cfg.min_contact_span,speed=s['relative_speed']<=self.grasp_cfg.hold_speed,off_ground=s['off_ground'],joint=s['joint_violation']<=self.grasp_cfg.max_joint_violation,valid=s['valid_hold'])
   for k,v in gates.items():
    p['gates'][k]=p['gates'].get(k,0)+int(v)
    if s['hand_contact']:p['contact_gates'][k]=p['contact_gates'].get(k,0)+int(v)
   if s['distance']<.18:p['near_ticks']+=1;p['near_speed_sum']+=s['relative_speed']
   if s['hand_contact'] and p['first_contact'] is None:
    p['first_contact']=dict(time=t,relative_speed=s['relative_speed'],distance=s['distance'],position=self.Dcmm.data.body('object').xpos.tolist(),local=s['local_ball'],palm_dot=s['palm_dot'])
 return s
def step(self,*a,**kw):
 obs,r,term,trunc,info=orig_step(self,*a,**kw)
 if term or trunc:info['bounce_grasp_probe']=self._probe
 return obs,r,term,trunc,info
BounceGraspEnv.reset=reset;BounceGraspEnv._grasp_state=state;BounceGraspEnv.step=step
# Exact existing evaluator, except identical draws are permitted for the zero-range ablation.
source=(ROOT/'scripts/_eval_multitarget_worker.py').read_text().replace("assert len({d['size_scale'] for d in draws})==n","assert len(draws)==n")
exec(compile(source,str(ROOT/'scripts/_eval_multitarget_worker.py'),'exec'),{'__name__':'__main__','__file__':str(ROOT/'scripts/_eval_multitarget_worker.py')})
