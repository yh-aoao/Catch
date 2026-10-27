"""Opt-in converged-contact candidate. No reward or controller-rate changes."""
import gymnasium as gym
import mujoco
import numpy as np
import random

VERSION = 'stable_v1'
TIMESTEP = .000125
SUBSTEPS = 16  # keep the original 2 ms force/controller update interval
SOLREF = (.008, .1)  # damping ratio is NOT a measured coefficient of restitution


class StableBounce(gym.Wrapper):
    @property
    def spec(self):
        return self.env.spec

    @spec.setter
    def spec(self, value):
        self.env.spec = value

    def __init__(self, env):
        super().__init__(env)
        self.apply()

    def apply(self):
        env = self.env.unwrapped
        model = env.Dcmm.model
        model.opt.timestep = TIMESTEP
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
        env._bounce_physics_substeps = SUBSTEPS
        env.fps = 1 / (env.steps_per_policy * TIMESTEP * SUBSTEPS)
        for gid in (env.object_id, env.floor_id):
            model.geom_solref[gid] = SOLREF
            model.geom_solimp[gid] = [.95, .998, .002, .5, 2.]
        mujoco.mj_forward(model, env.Dcmm.data)

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError('stable_v1 does not support reset options')
        # Frozen environments expose reset() without Gymnasium seed/options.
        # Their randomization uses NumPy's and Python's process-local RNGs.
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        obs, info = self.env.reset()
        # Legacy and YCB resets restore their own contact parameters each episode.
        self.apply()
        info['bounce_physics_version'] = VERSION
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['bounce_physics_version'] = VERSION
        return obs, reward, terminated, truncated, info


def make_bounce_stable(test_object='default', bounce_physics=VERSION, **kwargs):
    if bounce_physics != VERSION or kwargs.get('object_motion') != 'bounce':
        raise ValueError('StableBounce requires object_motion=bounce and bounce_physics=stable_v1')
    if kwargs.get('bounce_launch', 'legacy') != 'legacy' or kwargs.get('object_eval', False):
        raise ValueError('stable_v1 requires bounce_launch=legacy and object_eval=false')
    kwargs['bounce_physics'] = 'legacy'
    if test_object != 'default':
        from gym_dcmm.envs.object_test import make_object_test
        env = make_object_test(test_object, **kwargs)
    else:
        from gym_dcmm.envs.bounce_compat import make_bounce_july22, make_bounce_5fe75d5f
        factory = make_bounce_july22 if kwargs.get('task') == 'Tracking' else make_bounce_5fe75d5f
        env = factory(kwargs)
    return StableBounce(env)
