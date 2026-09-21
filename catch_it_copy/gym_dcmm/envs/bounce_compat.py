"""Compatibility entry; executable Bounce environment is in DcmmVecEnv.py."""


def make_bounce_5fe75d5f(parameters):
    from gym_dcmm.envs.DcmmVecEnv import _make_main_bounce
    return _make_main_bounce(parameters)
