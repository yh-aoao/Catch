"""Dispatch to the frozen 3a36f5f baseline without changing its methods."""


def make_bounce_3a36f5f(parameters):
    parameters = dict(parameters)
    for key in ('bounce_physics', 'bounce_launch'):
        if parameters.pop(key, 'legacy') != 'legacy':
            raise ValueError(
                'Bounce now reproduces 3a36f5f; P/L presets are not supported. '
                'Remove bounce_physics/bounce_launch overrides or use legacy.')
    verbose = parameters.pop('bounce_log', False)
    parameters.pop('basket_log', None)
    parameters.pop('roll_log', None)
    parameters['object_motion'] = 'bounce'
    from gym_dcmm.envs.DcmmVecEnv_bounce_3a36f5f import DcmmVecEnv
    env = DcmmVecEnv(**parameters)
    if verbose:
        print('[bounce-baseline] commit=3a36f5fb29c534db7639bfb9372458595b240edd '
              'environment=original; later bounce diagnostics are disabled', flush=True)
    return env
