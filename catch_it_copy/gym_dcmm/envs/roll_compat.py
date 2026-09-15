"""Route Roll to the frozen September 11 morning baseline."""


def make_roll_645edc4(parameters):
    parameters = dict(parameters)
    verbose = parameters.pop('roll_log', False)
    parameters['object_motion'] = 'roll'
    from gym_dcmm.envs.DcmmVecEnv_roll_645edc4 import DcmmVecEnv
    env = DcmmVecEnv(**parameters)
    if verbose:
        print('[roll-baseline] commit=645edc4e21631683254fba51cc66b035b409518d '
              'environment=original; execution_guard_v1 is not active', flush=True)
    return env
