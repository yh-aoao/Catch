"""Mobile throwing with a selectable stopping region around the trained distance."""
import numpy as np
from gym_dcmm.envs.basket_mixed import BasketMixedEnv
from gym_dcmm.utils.basket_tracking import limit_speed


def options(values):
    values=dict(values or {})
    if set(values)-{'y_offset_range'}:raise ValueError('Unknown free-stop options')
    bounds=np.asarray(values.get('y_offset_range',[-.08,.08]),dtype=float)
    if bounds.shape!=(2,) or not np.isfinite(bounds).all() or bounds[0]>=bounds[1]:
        raise ValueError('Free-stop y_offset_range must be a finite increasing pair')
    return dict(y_offset_range=bounds.tolist())


def region_state(nominal,base_position,yaw,cfg,values):
    """Reward/control geometry targets the nearest point in a corridor.
    Keep nominal observation coordinates for the inherited throwing network.
    """
    result=dict(nominal)
    delta=np.array(nominal['error_world'],copy=True)
    low,high=values['y_offset_range']
    offset=float(delta[1])
    delta[1]-=np.clip(delta[1],low,high)
    c,s=np.cos(yaw),np.sin(yaw)
    error=np.array([[c,s],[-s,c]])@delta
    target=np.array(nominal['target'],copy=True)
    target[:2]=np.asarray(base_position)[:2]+delta
    inside=bool(abs(delta[0])<cfg.basket_track_position_tolerance and low<=offset<=high)
    result.update(target=target,error_world=delta,distance=float(np.linalg.norm(delta)),
        desired_velocity=np.zeros(2) if inside else limit_speed(
            cfg.basket_track_slowdown_gain*error,cfg.basket_track_max_speed),
        in_position=inside,settled=inside and nominal['speed']<cfg.basket_track_speed_tolerance,
        free_stop=True,throw_distance_offset=offset)
    return result


class BasketFreeStopEnv(BasketMixedEnv):
    def __init__(self,basket_free_stop=None,**kwargs):
        self.free_stop_options=options(basket_free_stop)
        if kwargs.get('basket_reward_version')!='basket_mobile_v1' or kwargs.get('basket_mixed_ball')!='mixed':
            raise ValueError('Free stopping requires the conditional three-ball mobile task')
        mobile=kwargs.get('basket_mobile_options',{})
        if mobile.get('hold_position') or mobile.get('approach_braking'):
            raise ValueError('Free stopping requires policy braking; disable fixed position/approach brakes')
        super().__init__(**kwargs)

    def _basket_parking_state(self):
        nominal=super()._basket_parking_state()
        # Rehearsal stays at the original calibrated fixed position.
        if getattr(self,'basket_rehearsal_episode',False):return nominal
        rotation=self.Dcmm.data.body('base_link').xmat.reshape(3,3)
        yaw=np.arctan2(rotation[1,0],rotation[0,0])
        return region_state(nominal,self.Dcmm.data.body('arm_base').xpos,yaw,
            self.basket_cfg,self.free_stop_options)

    def step(self,action):
        obs,reward,done,truncated,info=super().step(action)
        state=self._basket_parking_state()
        info['basket_free_stop']=True
        info['basket_throw_distance_offset']=state.get('throw_distance_offset',0.)
        return obs,reward,done,truncated,info
