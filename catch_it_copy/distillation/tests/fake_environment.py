"""Tiny process-level fixture, never used by production configs."""
import numpy as np


class Environment:
    act_c_dim = 20
    obs_c_dim = 30

    def __init__(self, **kwargs):
        self.episode = 0

    def reset(self):
        self.episode += 1
        return {'episode': np.array([self.episode]), 'random': np.random.random(1)}, {}

    def step(self, action):
        if action['base'][0] < 0:
            raise ValueError('intentional worker error')
        return {'episode': np.array([-1]), 'random': np.zeros(1)}, 1., True, False, {'success': True}

    def close(self):
        pass
