"""Task factories; privileged teacher state never enters the student's forward inputs."""
from gym_dcmm.envs.basket_free_stop import BasketFreeStopEnv

class AutonomousBasketEnv(BasketFreeStopEnv):
    basket_autonomous_control=True


def make_catch(throw_shape=None,**kwargs):
    motion=kwargs['object_motion']
    if motion in ('roll','bounce'):
        from gym_dcmm.envs.object_benchmark import make_object_benchmark
        return make_object_benchmark(**kwargs)
    from gym_dcmm.envs.DcmmVecEnv import DcmmVecEnv
    from configs.env import DcmmCfg
    original=DcmmCfg.object_shape
    if throw_shape is not None:
        if throw_shape not in original:raise ValueError('Unknown Throw shape')
        DcmmCfg.object_shape=[throw_shape]
    try:return DcmmVecEnv(**kwargs)
    finally:DcmmCfg.object_shape=original

