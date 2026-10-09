import numpy as np


class Environment:
    act_t_dim = 8
    obs_t_dim = 18

    def __init__(self, **kwargs):
        assert kwargs['task'] == 'Tracking'
        assert kwargs['object_motion'] == 'bounce'
        self.episode = 0
        self.step_touch = False

    def reset(self):
        self.episode += 1
        self.step_touch = False
        return {'episode': np.array([self.episode])}, {}

    def step(self, action):
        if action['base'][0] < 0:
            raise ValueError('intentional worker error')
        self.step_touch = True
        return {'episode': np.array([-1])}, 1., False, True, {}

    def close(self):
        pass
