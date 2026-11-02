"""Opt-in ballistic guidance for learning fixed-base throws from scratch."""
import math
import numpy as np
from gym_dcmm.utils.basket_aim import crossing_error

DEFAULTS=dict(aim_sigma=.35,progress_bonus=8.,miss_cost=8.,
              launch_scale=.4,flight_scale=.6)


def make_options(options=None):
    options=dict(options or {})
    if set(options)-set(DEFAULTS):raise ValueError('Unknown Basket delivery option')
    result=dict(DEFAULTS,**options)
    for key,value in result.items():
        if not math.isfinite(float(value)) or float(value)<0:raise ValueError(key+' must be finite and nonnegative')
        result[key]=float(value)
    if result['aim_sigma']<=0:raise ValueError('aim_sigma must be positive')
    if result['launch_scale']>1 or result['flight_scale']>1:raise ValueError('Reward scales must be <=1')
    return result


def quality(position,velocity,center,gravity,tilt,options):
    position,velocity,center=map(lambda x:np.asarray(x,dtype=float),(position,velocity,center))
    if not all(np.isfinite(x).all() for x in (position,velocity,center)):return 0.,None
    error=crossing_error(position,velocity,center,gravity,tilt)
    if error is None:return 0.,None
    direction=center[:2]-position[:2];direction/=max(float(np.linalg.norm(direction)),1e-9)
    toward=float(np.clip(velocity[:2]@direction/.3,0.,1.))
    return toward*float(np.exp(-(error/options['aim_sigma'])**2)),float(error)


def new_state(initial_quality=0.):
    return dict(best=float(initial_quality),quality=float(initial_quality),pending=0.,miss_paid=False)


def observe(state,value,error,options):
    # Supported physics samples only. Repeated swings/recontacts cannot repay a peak.
    value=float(np.clip(value,0.,1.))
    state['pending']+=options['progress_bonus']*max(0.,value-state['best'])
    state['best']=max(state['best'],value)
    state['quality']=value;state['error']=-1. if error is None else float(error)


def adjustments(state,terms,options,*,done,success,collision):
    q=state.get('release_quality',state.get('quality',0.))
    result={key:terms.get(key,0.)*q if not collision else 0.
            for key in ('carry_forward','carry_position','carry_release')}
    # Negative slip and safety penalties retain their original meaning.
    result['valid_launch']=terms.get('valid_launch',0.)*options['launch_scale']
    result['flight_progress']=terms.get('flight_progress',0.)*options['flight_scale']
    result['ballistic_progress']=state.pop('pending',0.) if not collision else 0.
    state['pending']=0.
    result['task_miss']=0.
    if done and not success and not state['miss_paid']:
        result['task_miss']=-options['miss_cost'];state['miss_paid']=True
    return result
