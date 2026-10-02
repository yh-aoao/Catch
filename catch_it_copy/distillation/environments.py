"""Spawned environment pools with explicit terminal info and per-episode seeds.

Old project environments expose reset() without seed/options. Each worker owns
one environment; seed NumPy/Python before reset rather than changing that API.
"""
import importlib
import multiprocessing as mp
import os
import random
import traceback
import numpy as np
from .teachers import ROOT


def stack_tree(items):
    if isinstance(items[0], dict):
        return {key: stack_tree([item[key] for item in items]) for key in items[0]}
    return np.stack(items)


def _worker(pipe, task, config, seed, viewer):
    env = None
    try:
        os.chdir(ROOT)
        np.random.seed(seed)
        random.seed(seed)
        module_name, class_name = config.get('env_factory',
            'gym_dcmm.envs.DcmmVecEnv:DcmmVecEnv').split(':')
        factory = getattr(importlib.import_module(module_name), class_name)
        kwargs = dict(config['env_kwargs'])
        kwargs.update(task='Catching', object_motion=task, viewer=viewer)
        env = factory(**kwargs)
        if env.act_c_dim != 20 or env.obs_c_dim != 30:
            raise ValueError('Environment must expose canonical 20D actions / 30D Catch observation')
        episode = 0

        def reset():
            nonlocal episode
            # Stable per-episode seeds make teacher/student evaluation comparable.
            np.random.seed(seed + episode * 100003)
            random.seed(seed + episode * 100003)
            episode += 1
            return env.reset()[0]

        pipe.send(('ok', reset()))
        while True:
            command, payload = pipe.recv()
            if command == 'close':
                break
            if command != 'step':
                raise ValueError('Unknown environment request')
            obs, reward, terminated, truncated, info = env.step(payload)
            # Preserve final info before reset; never pair terminal obs with next action.
            done = bool(terminated or truncated)
            info = dict(info)
            if done:
                info.setdefault('terminated_reason', getattr(env, 'terminated_reason', '') or 'unspecified')
            info['_terminated'] = bool(terminated)
            info['_truncated'] = bool(truncated)
            if done:
                obs = reset()
            pipe.send(('ok', (obs, float(reward), done, info)))
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        try:
            pipe.send(('error', traceback.format_exc()))
        except (EOFError, BrokenPipeError):
            pass
    finally:
        if env is not None:
            env.close()
        pipe.close()


class Pool:
    def __init__(self, task, config, count, seed, viewer=False):
        self.pipes, self.processes = [], []
        self.timeout = config.get('worker_timeout_seconds', 180)
        try:
            context = mp.get_context('spawn')
            for i in range(count):
                parent, child = context.Pipe()
                process = context.Process(target=_worker,
                    args=(child, task, config, seed + i * 1009, viewer), daemon=True)
                process.start()
                child.close()
                self.pipes.append(parent)
                self.processes.append(process)
            self.obs = stack_tree([self._receive(p) for p in self.pipes])
        except BaseException:
            self.close()
            raise

    def _receive(self, pipe):
        if not pipe.poll(self.timeout):
            raise TimeoutError('Environment worker timed out; inspect MuJoCo/IK process output')
        status, payload = pipe.recv()
        if status == 'error':
            raise RuntimeError('Environment worker failed:\n' + payload)
        return payload

    def step(self, actions):
        for i, pipe in enumerate(self.pipes):
            pipe.send(('step', {key: value[i] for key, value in actions.items()}))
        results = [self._receive(p) for p in self.pipes]
        obs, reward, done, info = zip(*results)
        self.obs = stack_tree(obs)
        return self.obs, np.asarray(reward), np.asarray(done), list(info)

    def close(self):
        for pipe in self.pipes:
            try:
                pipe.send(('close', None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        for pipe in self.pipes:
            pipe.close()


def terminal_success(info, metric):
    if metric == 'legacy_truncated':
        return bool(info['_truncated'])
    if metric not in info:
        raise ValueError('Missing configured terminal success field: ' + metric)
    return bool(info[metric])
