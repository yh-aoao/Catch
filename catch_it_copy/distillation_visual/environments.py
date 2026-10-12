"""Same physics/teachers as state distillation, independent real depth renderer.

The old render() can return empty arrays in headless mode or consume NumPy RNG
for depth noise. Disable that diagnostic path only on this environment instance.
Render the current simulation state once per policy observation, without advancing
physics or consuming the physics RNG. No object-centred camera or segmentation.
"""
import importlib
import multiprocessing as mp
import os
import random
import sys
import traceback
import numpy as np
from distillation.teachers import ROOT
from distillation.environments import Pool as BasePool, stack_tree, terminal_success
from .core import encode_depth


class DepthCamera:
    def __init__(self, env, config):
        import mujoco
        self.env, self.config, self.mujoco = env, config, mujoco
        self.renderer, self.model = None, None

    def capture(self):
        model, data = self.env.Dcmm.model, self.env.Dcmm.data
        if model is not self.model:
            self.close()
            camera = self.config['camera']
            if self.mujoco.mj_name2id(model, self.mujoco.mjtObj.mjOBJ_CAMERA, camera) < 0:
                raise ValueError('Camera not found in environment: ' + camera)
            self.renderer = self.mujoco.Renderer(model, height=self.config['height'], width=self.config['width'])
            self.renderer.enable_depth_rendering()
            self.model = model
        self.renderer.update_scene(data, camera=self.config['camera'])
        return encode_depth(self.renderer.render(), self.config['near'], self.config['far'])

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
        self.renderer, self.model = None, None


def _worker(pipe, task, config, seed, viewer):
    env, camera, window = None, None, None
    try:
        # Set before any MuJoCo/OpenGL import inside this spawned process.
        backend = config['vision']['gl_backend']
        if backend == 'auto':
            backend = 'glfw' if viewer or sys.platform != 'linux' else 'egl'
        os.environ['MUJOCO_GL'] = backend
        os.chdir(ROOT)
        np.random.seed(seed)
        random.seed(seed)
        module, name = config.get('env_factory', 'gym_dcmm.envs.DcmmVecEnv:DcmmVecEnv').split(':')
        factory = getattr(importlib.import_module(module), name)
        kwargs = dict(config['env_kwargs'])
        kwargs.update(task='Tracking' if config['stage'] == 'track' else 'Catching',
                      object_motion=task, viewer=False, render_mode=None,
                      render_per_step=False, imshow_cam=False)
        env = factory(**kwargs)
        # Suppress only unused legacy info['imgs']; the independent camera below
        # always renders actual depth even on headless servers.
        env.render = lambda: np.empty((0,), dtype=np.float32)
        expected = (8, 18) if config['stage'] == 'track' else (20, 30)
        actual = (env.act_t_dim, env.obs_t_dim) if config['stage'] == 'track' else (env.act_c_dim, env.obs_c_dim)
        if actual != expected:
            raise ValueError('Environment observation/action dimensions changed')
        camera = DepthCamera(env, config['vision'])
        episode = 0

        def reset():
            nonlocal episode
            np.random.seed(seed + episode * 100003)
            random.seed(seed + episode * 100003)
            episode += 1
            return env.reset()[0]

        obs = reset()
        if viewer:
            import mujoco.viewer
            window = mujoco.viewer.launch_passive(env.Dcmm.model, env.Dcmm.data)
        pipe.send(('ok', {'teacher_obs': obs, 'depth': camera.capture()}))
        while True:
            command, payload = pipe.recv()
            if command == 'close':
                break
            if command != 'step':
                raise ValueError('Unknown worker request')
            obs, reward, terminated, truncated, info = env.step(payload)
            done = bool(terminated or truncated)
            info = dict(info)
            info.pop('imgs', None)
            if done:
                info.setdefault('terminated_reason', getattr(env, 'terminated_reason', '') or 'unspecified')
            info['_terminated'], info['_truncated'] = bool(terminated), bool(truncated)
            if config['stage'] == 'track':
                info['track_touch'] = bool(getattr(env, 'step_touch', False))
                info['track_contact_success'] = info['track_touch'] and not bool(terminated)
            if window is not None:
                window.sync()
            if done:
                obs = reset()
            packet = {'teacher_obs': obs, 'depth': camera.capture()}
            pipe.send(('ok', (packet, float(reward), done, info)))
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        try:
            pipe.send(('error', traceback.format_exc() + '\nVisual distillation requires an actual OpenGL depth renderer. '
                       'On a headless NVIDIA server use --gl-backend egl. Do not replace images with zeros.'))
        except (EOFError, BrokenPipeError):
            pass
    finally:
        if window is not None:
            window.close()
        if camera is not None:
            camera.close()
        if env is not None:
            env.close()
        pipe.close()


class Pool(BasePool):
    def __init__(self, task, config, count, seed, viewer=False):
        self.pipes, self.processes = [], []
        self.timeout = config.get('worker_timeout_seconds', 180)
        try:
            context = mp.get_context('spawn')
            for i in range(count):
                parent, child = context.Pipe()
                process = context.Process(target=_worker,
                    args=(child, task, config, seed + i*1009, viewer), daemon=True)
                process.start()
                child.close()
                self.pipes.append(parent)
                self.processes.append(process)
            self.obs = stack_tree([self._receive(p) for p in self.pipes])
        except BaseException:
            self.close()
            raise
