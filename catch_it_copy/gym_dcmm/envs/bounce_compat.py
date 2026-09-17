"""Dispatch bounce to the frozen 5fe75d5f baseline without changing other modes."""


def make_bounce_5fe75d5f(parameters):
    parameters = dict(parameters)
    for key in ('bounce_physics', 'bounce_launch'):
        if parameters.pop(key, 'legacy') != 'legacy':
            raise ValueError(
                'Bounce now reproduces 5fe75d5f; P/L presets are not supported. '
                'Remove bounce_physics/bounce_launch overrides or use legacy.')
    verbose = parameters.pop('bounce_log', False)
    parameters.pop('basket_log', None)
    parameters.pop('roll_log', None)
    parameters['object_motion'] = 'bounce'
    from gym_dcmm.envs.DcmmVecEnv_bounce_5fe75d5f import DcmmVecEnv
    env = DcmmVecEnv(**parameters)
    if verbose:
        print('[bounce-baseline] commit=5fe75d5f1e151c1fdc686fbf8f104db3870b2844 '
              'environment=original; later bounce diagnostics are disabled', flush=True)
    return env
