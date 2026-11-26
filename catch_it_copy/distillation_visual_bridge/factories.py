"""Direct Bridge control with a physical 0.6 m diameter basket."""
import copy
import numpy as np
from distillation_visual_basket.factories import make_catch


def make_basket(options, **kwargs):
    from basket_mobile_bridge.environment import AutonomousTeacherEnv
    from configs.env import DcmmCfg
    class BridgeEnvironment(AutonomousTeacherEnv):
        def reset(self, *args, **kwargs):
            previous=DcmmCfg.basket_radius
            DcmmCfg.basket_radius=.3
            try:return super().reset(*args,**kwargs)
            finally:DcmmCfg.basket_radius=previous
        def step(self, command):
            if isinstance(command,dict):
                physical=np.concatenate([command['base'],command['arm'],command['hand']])
                command=physical/np.repeat([.3,.025,.15],[2,6,12])
            return super().step(command)
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
