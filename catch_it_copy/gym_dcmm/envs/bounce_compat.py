"""Dispatch bounce: Tracking → 本地 2026-07-22 (58.18)；Catching → GitHub 5fe75d5f (BounceEnv)."""


def make_bounce_5fe75d5f(parameters):
    """GitHub 基线（5fe75d5f），用于 Catching：主文件内联的 BounceEnv。"""
    from gym_dcmm.envs.DcmmVecEnv import _make_main_bounce
    return _make_main_bounce(parameters)


def make_bounce_july22(parameters):
    """本地 2026-07-22 (58.18) 冻结基线，用于 Tracking。"""
    parameters = dict(parameters)
    for key in ('bounce_physics', 'bounce_launch'):
        if parameters.pop(key, 'legacy') != 'legacy':
            raise ValueError(
                'Bounce now reproduces 2026-07-22 (58.18); P/L presets are not supported. '
                'Remove bounce_physics/bounce_launch overrides or use legacy.')
    parameters.pop('bounce_log', None)
    parameters.pop('basket_log', None)
    parameters.pop('roll_log', None)
    parameters['object_motion'] = 'bounce'
    from gym_dcmm.envs.DcmmVecEnv_bounce_july22 import DcmmVecEnv
    env = DcmmVecEnv(**parameters)
    return env
