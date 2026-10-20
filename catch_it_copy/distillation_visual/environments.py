"""Same physics/teachers as state distillation, independent real depth renderer.

The old render() can return empty arrays in headless mode or consume NumPy RNG
for depth noise. Disable that diagnostic path only on this environment instance.
Render the current simulation state once per policy observation, without advancing
physics or consuming the physics RNG. No object-centred camera or segmentation.
"""
import faulthandler
import importlib
import multiprocessing as mp
import os
import random
import sys
import traceback
import numpy as np
from distillation.teachers import ROOT
from distillation.environments import Pool as BasePool, stack_tree, terminal_success
from .core import encode_depth, camera_specs


class DepthCamera:
    def __init__(self, env, config):
        import mujoco
        self.env, self.config, self.mujoco = env, config, mujoco
        self.renderer, self.model = None, None

    def capture(self):
        return encode_depth(self.capture_metres(), self.config['near'], self.config['far'])

    def capture_metres(self, camera=None):
        camera = camera or self.config['camera']
        model, data = self.env.Dcmm.model, self.env.Dcmm.data
        if self.mujoco.mj_name2id(model, self.mujoco.mjtObj.mjOBJ_CAMERA, camera) < 0:
            raise ValueError('Camera not found in environment: ' + camera)
        if model is not self.model:
            self.close()
            print('[depth-renderer] pid={} backend={} EGL_DEVICE={} size={}x{} offscreen={}x{}'.format(
                os.getpid(), os.environ.get('MUJOCO_GL'), os.environ.get('MUJOCO_EGL_DEVICE_ID', 'default'),
                self.config['width'], self.config['height'],
                model.vis.global_.offwidth, model.vis.global_.offheight), flush=True)
            self.renderer = self.mujoco.Renderer(model, height=self.config['height'], width=self.config['width'])
            self.renderer.enable_depth_rendering()
            self.model = model
        self.renderer.update_scene(data, camera=camera)
        depth = np.asarray(self.renderer.render(), dtype=np.float32).copy()
        if depth.ndim != 2 or not np.isfinite(depth).all() or (depth < 0).any():
            raise ValueError('Invalid metric depth from renderer')
        return depth

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
        self.renderer, self.model = None, None


def _worker(pipe, task, config, seed, viewer):
    env, window = None, None
    sensor = None
    try:
        faulthandler.enable()
        # Set before any MuJoCo/OpenGL import inside this spawned process.
        backend = config['vision']['gl_backend']
        if backend == 'auto':
            backend = 'glfw' if viewer or sys.platform != 'linux' else 'egl'
        os.environ['MUJOCO_GL'] = backend
        os.chdir(ROOT)
        np.random.seed(int(seed) % (2**32))
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
        specs = camera_specs(config['vision'])
        names = list(dict.fromkeys([s['name'] for s in specs] + config.get('inspection_cameras', [])))
        sensor = DepthCamera(env, dict(config['vision'], camera=names[0]))
        episode = 0

        def packet(obs):
            before = float(env.Dcmm.data.time)
            # No env.step(), reset(), mj_step() or viewer synchronization between views.
            images = {name: sensor.capture_metres(name) for name in names}
            after = float(env.Dcmm.data.time)
            if after != before:
                raise RuntimeError('Simulation advanced between camera captures')
            result = {'teacher_obs': obs,
                      'depth': np.concatenate([encode_depth(images[s['name']], s['near'], s['far'])
                                               for s in specs], axis=0),
                      'sim_time': before, 'episode_id': episode}
            if config.get('inspection_cameras'):
                result['camera_depth_metres'] = images
            return result

        def reset():
            nonlocal episode
            np.random.seed((int(seed) + int(episode) * 100003) % (2**32))
            random.seed(seed + episode * 100003)
            episode += 1
            return env.reset()[0]

        obs = reset()
        if viewer:
            import mujoco.viewer
            window = mujoco.viewer.launch_passive(env.Dcmm.model, env.Dcmm.data)
        pipe.send(('ok', packet(obs)))
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
            pipe.send(('ok', (packet(obs), float(reward), done, info)))
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        try:
            pipe.send(('error', traceback.format_exc()))
        except (EOFError, BrokenPipeError):
            pass
    finally:
        if window is not None:
            window.close()
        if sensor is not None:
            sensor.close()
        if env is not None:
            env.close()
        pipe.close()


class Pool(BasePool):
    def __init__(self, task, config, count, seed, viewer=False):
        self.pipes, self.processes = [], []
        self.task = task
        self.timeout = config.get('worker_timeout_seconds', 180)
        try:
            # Inherited before child imports NumPy/Torch; respect explicit settings.
            for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
                os.environ.setdefault(name, '1')
            context = mp.get_context('spawn')
            observations = []
            for i in range(count):
                parent, child = context.Pipe()
                process = context.Process(target=_worker,
                    args=(child, task, config, seed + i*1009, viewer), daemon=True)
                process.start()
                child.close()
                self.pipes.append(parent)
                self.processes.append(process)
                # Avoid simultaneous MuJoCo/EGL initialization spikes. Runtime stays parallel.
                observations.append(self._receive(parent))
            self.obs = stack_tree(observations)
        except BaseException:
            self.close()
            raise


    def _receive(self, pipe):
        index = self.pipes.index(pipe)
        process = self.processes[index]
        try:
            if not pipe.poll(self.timeout):
                raise TimeoutError('Environment worker timed out: task={} worker={} pid={} exitcode={}'.format(
                    self.task, index, process.pid, process.exitcode))
            status, payload = pipe.recv()
        except (EOFError, OSError) as error:
            process.join(timeout=0.5)
            raise RuntimeError(
                'Environment worker exited without a response: task={} worker={} pid={} exitcode={}. '
                'On Linux -9 means SIGKILL (check host/container RAM limits and OOM logs; '
                'it does not alone prove OOM); -11 means SIGSEGV; -6 means SIGABRT. '
                'Inspect preceding child stderr for MuJoCo/EGL errors. '
                'Retry with --num-envs 1 --gl-backend egl and a new output directory. '
                'num-envs is PER TASK, so 8 means 24 workers across three tasks.'
                .format(self.task, index, process.pid, process.exitcode)) from error
        if status == 'error':
            raise RuntimeError('Environment worker failed: task={} worker={} pid={}\n{}'.format(
                self.task, index, process.pid, payload))
        if status != 'ok':
            raise RuntimeError('Unknown environment worker response: ' + str(status))
        return payload
