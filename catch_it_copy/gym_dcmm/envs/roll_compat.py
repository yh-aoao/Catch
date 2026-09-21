"""Route Roll to the frozen September 11 morning baseline."""


def make_roll_645edc4(parameters):
    parameters = dict(parameters)
    verbose = parameters.pop('roll_log', False)
    parameters['object_motion'] = 'roll'
    tracking = parameters.get('task') == 'Tracking'
    if tracking:
        from gym_dcmm.envs.DcmmVecEnv_roll_track_b66666c import DcmmVecEnv
    else:
        from gym_dcmm.envs.DcmmVecEnv_roll_645edc4 import DcmmVecEnv
    env = DcmmVecEnv(**parameters)
    env.roll_log = verbose
    if verbose:
        import inspect
        if tracking:
            import configs.env.DcmmCfg_roll_track_b66666c as cfg
        else:
            import configs.env.DcmmCfg_roll_645edc4 as cfg
        print('[roll-baseline] task={} revision={}'.format(
            parameters.get('task'), 'b66666c' if tracking else 'current_catching'), flush=True)
        print('[roll-baseline-source] environment={} config={}'.format(
            inspect.getfile(DcmmVecEnv), cfg.__file__), flush=True)
    return env
