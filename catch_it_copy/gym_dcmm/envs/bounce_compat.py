"""Dispatch to the frozen 61709fc baseline without changing its methods."""


def make_bounce_61709fc(parameters):
    parameters = dict(parameters)
    for key in ('bounce_physics', 'bounce_launch'):
        if parameters.pop(key, 'legacy') != 'legacy':
            raise ValueError(
                'Bounce now reproduces 61709fc; P/L presets are not supported. '
                'Remove bounce_physics/bounce_launch overrides or use legacy.')
    verbose = parameters.pop('bounce_log', False)
    parameters.pop('basket_log', None)
    parameters.pop('roll_log', None)
    parameters['object_motion'] = 'bounce'
    from gym_dcmm.envs.DcmmVecEnv_bounce_61709fc import DcmmVecEnv
    env = DcmmVecEnv(**parameters)
    if verbose:
        print('[bounce-baseline] commit=61709fc82bb4e48aa6a95fb722874ddf7ab483d0 '
              'environment=original; later bounce diagnostics are disabled', flush=True)
    return env
