"""Direct Bridge control with a physical 0.6 m diameter basket."""
import copy
import numpy as np
from distillation_visual_basket.factories import make_catch


def make_basket(options, **kwargs):
    from basket_mobile_bridge.environment import AutonomousTeacherEnv
    from configs.env import DcmmCfg
    class BridgeEnvironment(AutonomousTeacherEnv):
        def reset(self, *args, **kwargs):
            self._region_hit=False;self._stopped_region_hit=False
            previous=DcmmCfg.basket_radius
            DcmmCfg.basket_radius=.3
            try:return super().reset(*args,**kwargs)
            finally:DcmmCfg.basket_radius=previous
        def _basket_check_flight(self,previous):
            super()._basket_check_flight(previous)
            if self._initializing:return
            release=self._release
            if not (self._released_once and release and release.get('confirmed',False) and not self.touching()):return
            from basket_autonomous.carry_reward import purposeful_launch
            if not purposeful_launch(release['position'],release['velocity'],self.basket_center,release['predicted_error']):return
            from gym_dcmm.utils.basket_catching import hoop_crossing
            crossed,hit,_=hoop_crossing(previous,self.Dcmm.data.qpos[37:40],self.basket_center,
                self.basket_cfg.basket_tilt_deg,self.basket_cfg.basket_radius,self.radius)
            if crossed and hit:
                self._region_hit=True
                self._stopped_region_hit|=bool(release['valid_stop'])

        def step(self, command):
            if isinstance(command,dict):
                physical=np.concatenate([command['base'],command['arm'],command['hand']])
                command=physical/np.repeat([.3,.025,.15],[2,6,12])
            obs,reward,done,trunc,info=super().step(command)
            safe=not (self.basket_robot_collision or self.basket_arm_self_collision)
            info.update(basket_region_hit=bool(self._region_hit and safe),
                        basket_stopped_region_hit=bool(self._stopped_region_hit and safe))
            return obs,reward,done,trunc,info
    options=copy.deepcopy(options)
    diameter=float(options.pop('basket_diameter_m'))
    if diameter!=.6:raise ValueError('Expected a 0.6 m diameter rim')
    original=DcmmCfg.basket_radius
    DcmmCfg.basket_radius=diameter/2
    try:
        env=BridgeEnvironment(options,img_size=(160,160),print_obs=False,print_info=False)
    finally:
        DcmmCfg.basket_radius=original
    if not np.isclose(env.basket_cfg.basket_radius,.3):raise ValueError('Basket radius mismatch')
    return env
