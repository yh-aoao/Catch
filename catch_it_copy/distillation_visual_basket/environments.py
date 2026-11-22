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
from .core import encode_depth, camera_specs, robot_input, command_context
from visual_module_experiments.labels import labels


def hand_contact(env):
    """Include analytical YCB collision spheres, not only their visual 'object' geom."""
    from gym_dcmm.utils.roll_rewards import hand_collision_ids
    model,data=env.Dcmm.model,env.Dcmm.data
    objects=set(np.flatnonzero(model.geom_bodyid==model.body('object').id).tolist())
    hand=set(hand_collision_ids(model,env.hand_start_id).tolist())
    return any((int(c.geom1) in objects and int(c.geom2) in hand) or
               (int(c.geom2) in objects and int(c.geom1) in hand) for c in data.contact)


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
        import torch
        torch.set_num_threads(1)
        os.chdir(ROOT)
        np.random.seed(int(seed) % (2**32))
        random.seed(seed)
        module, name = config.get('env_factory', 'gym_dcmm.envs.DcmmVecEnv:DcmmVecEnv').split(':')
        factory = getattr(importlib.import_module(module), name)
        kwargs = dict(config['env_kwargs'])
        kwargs.update(task='Tracking' if config['stage'] == 'track' else 'Catching',
                      object_motion='throw_basket' if task=='basket' else task, viewer=False, render_mode=None,
                      render_per_step=False, imshow_cam=False)
        env = factory(**kwargs)
        # Suppress only unused legacy info['imgs']; the independent camera below
        # always renders actual depth even on headless servers.
        env.unwrapped.render = lambda: np.empty((0,), dtype=np.float32)
        expected = (20, 51) if task=='basket' else (20,30)
        actual = (env.act_c_dim, env.obs_c_dim)
        if actual != expected:
            raise ValueError('Environment observation/action dimensions changed')
        specs = camera_specs(config['vision'])
        names = list(dict.fromkeys([s['name'] for s in specs] + config.get('inspection_cameras', [])))
        # One GL context per environment; views are rendered sequentially without stepping.
        sensor = DepthCamera(env.unwrapped, dict(config['vision'], camera=names[0]))
        contact_seen = False
        original_contacts = env.unwrapped._get_contacts
        def recorded_contacts():
            nonlocal contact_seen
            value = original_contacts()
            contact_seen = contact_seen or hand_contact(env)
            return value
        env.unwrapped._get_contacts = recorded_contacts
        episode = 0
        previous_object = None
        previous_time = None

        def packet(obs):
            nonlocal previous_object, previous_time
            before = float(env.Dcmm.data.time)
            # No env.step(), reset(), mj_step() or viewer synchronization between views.
            images = {name: sensor.capture_metres(camera=name) for name in names}
            after = float(env.Dcmm.data.time)
            if after != before:
                raise RuntimeError('Simulation advanced between camera captures')
            result = {'teacher_obs': obs,
                      'depth': np.concatenate([encode_depth(images[s['name']], s['near'], s['far'])
                                               for s in specs], axis=0),
                      'sim_time': before, 'episode_id': episode}
            position = np.asarray(env.unwrapped._get_relative_object_pos3d(), dtype=np.float32).copy()
            velocity = np.zeros(3, np.float32)
            velocity_valid = previous_time is not None and before > previous_time
            if velocity_valid:
                velocity = (position - previous_object) / (before - previous_time)
            target = np.concatenate([position, velocity])
            if target.shape != (6,) or not np.isfinite(target).all():
                raise ValueError('Invalid auxiliary object target')
            result['proprio'] = robot_input(obs, np.asarray(env.Dcmm.data.qpos[15:21],np.float32),
                                            np.asarray(env.Dcmm.data.qvel[14:20],np.float32))
            result['context'] = command_context(obs, task)
            result['basket_target'] = (np.asarray(env.Dcmm.data.body('arm_base').xmat.reshape(3,3).T@(env.basket_center-env.Dcmm.data.body('arm_base').xpos),np.float32)
                                       if task=='basket' else np.zeros(3,np.float32))
            result['basket_valid'] = np.float32(task=='basket')
            result['basket_phase'] = np.float32(env.unwrapped.basket_mobile_state['unlocked']) if task=='basket' else np.float32(0)
            if task=='basket':
                raw=env.unwrapped
                result['teacher_hold_arm'] = raw.basket_carry_arm.copy()
                result['teacher_hold_hand_delta'] = (raw.basket_carry_hand-raw.Dcmm.target_hand_qpos).copy()
                result['teacher_phase'] = np.float32(raw.basket_mobile_state['unlocked'])
                result['arm_measured'] = raw.Dcmm.data.qpos[15:21].astype(np.float32).copy()
                result['policy_dt'] = np.float32(raw.steps_per_policy*raw.Dcmm.model.opt.timestep)
                projections=[]
                for spec in specs:
                    camera=raw.Dcmm.model.camera(spec['name'])
                    local=raw.Dcmm.data.cam_xmat[camera.id].reshape(3,3).T@(raw.basket_center-raw.Dcmm.data.cam_xpos[camera.id])
                    half=np.tan(np.deg2rad(raw.Dcmm.model.cam_fovy[camera.id])/2)
                    projections.append(bool(local[2]<0 and abs(local[0])<-local[2]*half*config['vision']['width']/config['vision']['height'] and abs(local[1])<-local[2]*half))
                result['basket_center_in_camera_frustum']=np.asarray(projections,np.float32)
            result['object_name'] = config['selected_object']
            result['object_target'] = target
            result['object_valid'] = np.array([1, 1, 1] + [int(velocity_valid)] * 3, np.float32)
            if task=='basket':
                result['object_valid'][:] = 0
            result.update(labels(env.unwrapped, sensor, specs, False))
            previous_object, previous_time = position, before
            if config.get('inspection_cameras'):
                result['camera_depth_metres'] = images
            return result

        def reset():
            nonlocal episode, previous_object, previous_time
            previous_object, previous_time = None, None
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
            contact_seen = False
            obs, reward, terminated, truncated, info = env.step(payload)
            done = bool(terminated or truncated)
            info = dict(info)
            info['module_hand_contact'] = contact_seen or hand_contact(env)
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
    """Fixed object slots; each condition has independent physics and RNG."""
    def __init__(self,task,config,count,seed,viewer=False,objects=None):
        import copy
        self.task=task;self.pipes=[];self.processes=[]
        self.timeout=config.get('worker_timeout_seconds',180)
        names=objects or config['objects']
        if count<len(names):raise ValueError('Need at least one environment per object')
        try:
            for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'1')
            context=mp.get_context('spawn');observations=[]
            for i in range(count):
                cfg=copy.deepcopy(config);obj=names[i%len(names)];cfg['selected_object']=obj
                if task=='throw':cfg['env_kwargs']['throw_shape']=obj
                if task in ('roll','bounce'):
                    cfg['env_kwargs']['test_object']='default' if obj=='original' else 'ycb:'+obj
                parent,child=context.Pipe()
                process=context.Process(target=_worker,args=(child,task,cfg,seed+i*1009,viewer),daemon=True)
                process.start();child.close();self.pipes.append(parent);self.processes.append(process)
                observations.append(self._receive(parent))
            self.obs=stack_tree(observations)
        except BaseException:
            self.close();raise

    def step_async(self,actions):
        for i,pipe in enumerate(self.pipes):pipe.send(('step',{k:v[i] for k,v in actions.items()}))

    def step_wait(self):
        results=[self._receive(p) for p in self.pipes]
        obs,reward,done,info=zip(*results);self.obs=stack_tree(obs)
        return self.obs,np.asarray(reward),np.asarray(done),list(info)
