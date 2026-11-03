"""Play exact, precomputed Bounce policy trajectories at normal wall-clock speed.

Precomputation uses the trained physics, policy, normalization and 2 ms control
ticks. Playback restores recorded states and runs forward kinematics only. It
never changes the simulation timestep or integrates a second, approximate ball
trajectory. A cached recording can be replayed without running the policy.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import tempfile
import threading
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import mujoco
import numpy as np
from omegaconf import OmegaConf
import torch

from ycb_objects.evaluate_trained_balls import PolicyEnvironment, batched
from ycb_objects.generalization_probe import physical_parameters


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT/'outputs/bounce_three_ball_catching_v1_seed123/2026-10-28/11:36:32/nn/bounce_catch_best_reward_464.31.pth'
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
MODEL_ARRAYS = ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat',
    'body_pos', 'body_quat', 'geom_size', 'geom_pos', 'geom_quat', 'geom_rgba',
    'geom_friction', 'geom_solref', 'geom_solimp', 'geom_contype', 'geom_conaffinity',
    'dof_armature', 'dof_damping', 'jnt_range', 'jnt_solref', 'jnt_solimp',
    'pair_friction', 'pair_solref', 'pair_solimp', 'actuator_ctrlrange')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_signature(model):
    """Parameters needed for the same physical and visual scene across seeds."""
    digest = hashlib.sha256()
    for name in MODEL_ARRAYS:
        digest.update(name.encode())
        digest.update(np.asarray(getattr(model, name)).tobytes())
    digest.update(np.asarray([model.opt.timestep, int(model.opt.integrator)]).tobytes())
    return digest.hexdigest()


def frame_index(times, elapsed):
    """Choose a real recorded state, with at most one control tick of delay."""
    return int(np.clip(np.searchsorted(times-times[0], elapsed, side='right')-1,
                       0, len(times)-1))


def restore_frame(model, data, trace, index):
    mujoco.mj_setState(model, data, trace['state'][index], STATE_SPEC)
    mujoco.mj_forward(model, data)
    # These assertions also make replay verification possible without OpenGL.
    np.testing.assert_array_equal(data.qpos, trace['qpos'][index])
    np.testing.assert_array_equal(data.qvel, trace['qvel'][index])


@contextmanager
def joined_passive_viewer(model, data, key_callback=None, cleanup_stats=None):
    """Wait for the public passive viewer's UI thread to finish on close.

    MuJoCo 3.6 Handle.close() requests exit but does not join the daemon UI
    thread created by launch_passive(). Letting Python exit before that thread
    destroys its GL context can race glfw.terminate() during atexit. This
    standalone application records the threads created by its public launch
    call and joins them after public close(), while model/data are still alive.
    No private viewer attributes or forced process exits are used.
    """
    import mujoco.viewer

    before = set(threading.enumerate())
    viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)
    created = [thread for thread in threading.enumerate()
               if thread not in before and thread is not threading.current_thread()]
    try:
        yield viewer
    finally:
        viewer.close()
        for thread in created:
            thread.join()
        if cleanup_stats is not None:
            cleanup_stats.update(close_requested=True, joined_threads=len(created),
                thread_names=[thread.name for thread in created],
                threads_alive_after_join=sum(thread.is_alive() for thread in created))


@torch.inference_mode()
def precompute(checkpoint, config_path, object_name, episodes, seed, output):
    from gym_dcmm.envs.bounce_physical_v2 import make_bounce_physical_v2, DT, SUBSTEPS

    checkpoint, config_path, output = Path(checkpoint), Path(config_path), Path(output)
    if not checkpoint.is_file() or not config_path.is_file():
        raise FileNotFoundError('Checkpoint and its resolved_config.yaml must exist')
    cfg = OmegaConf.load(config_path)
    if cfg.task != 'Catching_TwoStage' or cfg.object_motion != 'bounce' or cfg.bounce_physics != 'physical_v2':
        raise ValueError('This viewer supports physical_v2 Bounce Catching_TwoStage checkpoints')
    version = str(cfg.get('bounce_grasp_version', 'legacy'))
    if version == 'legacy':
        from gym_dcmm.algs.ppo_dcmm_5fe75d5f.ppo_dcmm_catch_two_stage import PPO_Catch_TwoStage as Agent
    elif version == 'controlled_hold_v1':
        from gym_dcmm.algs.ppo_dcmm_5fe75d5f.ppo_dcmm_catch_hold import PPO_Catch_Hold as Agent
    else:
        raise ValueError('Unsupported Bounce grasp version: '+version)
    cfg.test, cfg.num_envs, cfg.train.ppo.num_actors = True, 1, 1
    cfg.rl_device, cfg.device_id = 'cpu', -1
    cfg.viewer = cfg.imshow_cam = False
    cfg.checkpoint_tracking = cfg.checkpoint_catching = ''
    material = ROOT/'ycb_objects/calibration_data/balls.three_ball_training_v1.json'
    cfg.bounce_material_file = str(material)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    name = 'default' if object_name == 'original' else 'ycb:'+object_name
    env = make_bounce_physical_v2(task='Catching', object_motion='bounce',
        test_object=name, bounce_launch=cfg.bounce_launch,
        bounce_v2=OmegaConf.to_container(cfg.bounce_v2), bounce_material_file=material,
        bounce_grasp_version=version,
        bounce_grasp=OmegaConf.to_container(cfg.bounce_grasp) if cfg.get('bounce_grasp') else None,
        viewer=False, render_mode=None, object_eval=False, env_time=2.5, steps_per_policy=20)
    agent = None
    episode_meta = []
    initial_signature = None
    started = time.monotonic()
    model, data = env.Dcmm.model, env.Dcmm.data
    state_size = mujoco.mj_stateSize(model, STATE_SPEC)
    active = False
    snapshots, qposes, qvels, times = [], [], [], []

    def capture():
        state = np.empty(state_size)
        mujoco.mj_getState(model, data, state, STATE_SPEC)
        snapshots.append(state)
        qposes.append(data.qpos.copy())
        qvels.append(data.qvel.copy())
        times.append(float(data.time))

    original_contacts = env._get_contacts

    def record_contacts():
        result = original_contacts()
        if active:
            capture()
        return result

    env._get_contacts = record_contacts
    try:
        with tempfile.TemporaryDirectory(prefix='bounce_replay_agent_') as temporary:
            agent = Agent(PolicyEnvironment(env), temporary, cfg)
            agent.restore_test(str(checkpoint))
            raw = torch.load(checkpoint, map_location='cpu')
            agent.model.load_state_dict(raw['model'], strict=True)
            agent.set_eval()
            for episode in range(episodes):
                active = False
                torch.manual_seed(seed+episode)
                obs, _ = env.reset(seed=seed+episode)
                if model.opt.timestep != DT or env._bounce_physics_substeps != SUBSTEPS:
                    raise RuntimeError('Replay must precompute with the trained baseline physics')
                signature = model_signature(model)
                if initial_signature is None:
                    initial_signature = signature
                    mujoco.mj_saveModel(model, str(output/'scene.mjb'))
                elif signature != initial_signature:
                    raise RuntimeError('Scene parameters changed across episodes; use a separate recording')
                snapshots, qposes, qvels, times = [], [], [], []
                capture()
                active = True
                total_reward = 0.
                for step in range(70):
                    action = agent.model_act({'obs':agent.obs2tensor(batched(obs))},
                                             inference=True)['actions'].clamp(-1, 1)
                    action = torch.nn.functional.pad(action, (0, agent.full_action_dim-action.shape[1]))
                    commands = agent.action2dict(action)
                    obs, reward, terminated, truncated, info = env.step({k:v[0] for k,v in commands.items()})
                    total_reward += float(reward)
                    if terminated or truncated:
                        break
                else:
                    raise RuntimeError('Episode exceeded the 2.5s evaluation horizon')
                active = False
                terminal_state = np.empty(state_size)
                mujoco.mj_getState(model, data, terminal_state, STATE_SPEC)
                # No state changes between the final controller sample and
                # the environment's true terminal state may be silently lost.
                np.testing.assert_array_equal(snapshots[-1], terminal_state)
                np.testing.assert_array_equal(qposes[-1], data.qpos)
                np.testing.assert_array_equal(qvels[-1], data.qvel)
                trace = dict(time=np.asarray(times), state=np.asarray(snapshots),
                             qpos=np.asarray(qposes), qvel=np.asarray(qvels))
                assert np.isfinite(trace['state']).all()
                deltas = np.diff(trace['time'])
                np.testing.assert_allclose(deltas, DT*SUBSTEPS, atol=1e-10, rtol=0)
                filename = 'episode_{:03d}.npz'.format(episode)
                np.savez_compressed(output/filename, **trace)
                entry = dict(index=episode, seed=seed+episode, trace=filename,
                    trace_sha256=sha256(output/filename), frames=len(times),
                    simulation_seconds=float(times[-1]-times[0]),
                    success=bool(info.get('success')), success_version=info.get('success_version'),
                    reason=str(info.get('terminated_reason') or env.terminated_reason or 'timeout'),
                    policy_steps=step+1, reward=total_reward,
                    terminal_state_exact=True, model_signature=signature,
                    physics=physical_parameters(env))
                episode_meta.append(entry)
                print('[precompute] episode={}/{} seed={} sim={:.3f}s success={} reason={}'.format(
                    episode+1, episodes, seed+episode, entry['simulation_seconds'],
                    entry['success'], entry['reason']), flush=True)
    finally:
        if agent is not None:
            agent.writer.close()
        env.close()
    source_paths = ['ycb_objects/bounce_viewer_replay.py', 'gym_dcmm/envs/bounce_physical_v2.py',
                    'gym_dcmm/envs/DcmmVecEnv.py', 'gym_dcmm/agents/MujocoDcmm_bounce_5fe75d5f.py']
    manifest = dict(version='exact_bounce_replay_v1', mode='recorded_state_playback_not_live_policy',
        object=object_name, checkpoint=str(checkpoint.resolve()), checkpoint_sha256=sha256(checkpoint),
        config=str(config_path.resolve()), config_sha256=sha256(config_path),
        material=str(material), material_sha256=sha256(material),
        grasp_version=version, physics_dt_seconds=DT, controller_interval_seconds=DT*SUBSTEPS,
        physics_substeps=SUBSTEPS, policy_interval_seconds=DT*SUBSTEPS*20,
        state_spec=int(STATE_SPEC), model='scene.mjb', model_sha256=sha256(output/'scene.mjb'),
        model_signature=initial_signature, episodes=episode_meta,
        precompute_wall_seconds=time.monotonic()-started,
        mujoco_version=mujoco.__version__, torch_version=torch.__version__,
        effective_policy_config=OmegaConf.to_container(cfg, resolve=True),
        source_sha256={p:sha256(ROOT/p) for p in source_paths})
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    (output/'COMPLETE').write_text('exact precomputation complete\n')
    print('[precompute-complete] cache={} wall={:.2f}s total_sim={:.2f}s'.format(
        output, manifest['precompute_wall_seconds'],
        sum(e['simulation_seconds'] for e in episode_meta)), flush=True)
    return manifest


def load_recording(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest['version'] != 'exact_bounce_replay_v1' or not (directory/'COMPLETE').exists():
        raise ValueError('Not a complete exact Bounce recording')
    if manifest['state_spec'] != int(STATE_SPEC):
        raise ValueError('Recorded MuJoCo state specification does not match')
    if sha256(directory/manifest['model']) != manifest['model_sha256']:
        raise ValueError('Recorded compiled scene has changed')
    model = mujoco.MjModel.from_binary_path(str(directory/manifest['model']))
    if model_signature(model) != manifest['model_signature']:
        raise ValueError('Compiled scene parameters do not match the recording')
    traces = []
    for episode in manifest['episodes']:
        filename = directory/episode['trace']
        if sha256(filename) != episode['trace_sha256']:
            raise ValueError('Recorded state data has changed: '+str(filename))
        with np.load(filename) as stored:
            trace = {name:stored[name].copy() for name in ('time', 'state', 'qpos', 'qvel')}
        if len(trace['time']) != episode['frames'] or not np.all(np.diff(trace['time']) > 0):
            raise ValueError('Invalid recorded time sequence')
        traces.append(trace)
    return model, traces, manifest


def playback(directory, speed=1., fps=60., once=False, max_playback_seconds=None):
    model, traces, manifest = load_recording(directory)
    data = mujoco.MjData(model)
    keys = queue.SimpleQueue()
    stats = dict(speed=float(speed), wall_seconds=0., simulation_seconds=0.,
                 displayed_frames=0, episodes_started=1, mode='exact_recorded_states')
    paused = False
    index, playhead = 0, 0.
    print('[replay] {} episodes, speed={}x; Space=pause, R=restart, Left/Right=episode, Esc=quit'.format(
        len(traces), speed), flush=True)
    print('[replay] This is recorded-state playback; mouse movement changes the camera, not the recorded policy.', flush=True)
    restore_frame(model, data, traces[0], 0)
    window_started = started = last = time.monotonic()
    ended = None
    cleanup = {}
    try:
        with joined_passive_viewer(model, data, key_callback=keys.put,
                                   cleanup_stats=cleanup) as viewer:
            viewer.cam.lookat[:] = [0., .8, .55]
            viewer.cam.distance = 4.
            viewer.cam.azimuth = 180.
            viewer.cam.elevation = -15.
            viewer.sync()
            # Window/context creation is preparation, not replay time.
            started = last = time.monotonic()
            while viewer.is_running():
                frame_started = now = time.monotonic()
                if max_playback_seconds is not None and now-started >= max_playback_seconds:
                    break
                quit_requested = False
                while not keys.empty():
                    key = keys.get()
                    if key == 256:
                        quit_requested = True
                    elif key == 32:
                        paused = not paused
                    elif key in (82, 114):
                        playhead = 0.
                    elif key in (262, 263):
                        index = (index+(1 if key == 262 else -1)) % len(traces)
                        playhead = 0.
                        stats['episodes_started'] += 1
                if quit_requested:
                    break
                elapsed = max(0., now-last)
                last = now
                if not paused:
                    progress = elapsed*speed
                    playhead += progress
                    stats['simulation_seconds'] += progress
                duration = float(traces[index]['time'][-1]-traces[index]['time'][0])
                if playhead >= duration:
                    restore_frame(model, data, traces[index], len(traces[index]['time'])-1)
                    viewer.sync()
                    if once and index == len(traces)-1:
                        break
                    playhead = max(0., playhead-duration)
                    index = (index+1) % len(traces)
                    stats['episodes_started'] += 1
                restore_frame(model, data, traces[index], frame_index(traces[index]['time'], playhead))
                viewer.sync()
                stats['displayed_frames'] += 1
                delay = 1./fps-(time.monotonic()-frame_started)
                if delay > 0:
                    time.sleep(delay)
            ended = time.monotonic()
    finally:
        stats['wall_seconds'] = (ended if ended is not None else time.monotonic())-started
        stats['window_total_wall_seconds'] = time.monotonic()-window_started
        stats['observed_simulation_per_wall'] = stats['simulation_seconds']/max(stats['wall_seconds'], 1e-9)
        stats['displayed_fps'] = stats['displayed_frames']/max(stats['wall_seconds'], 1e-9)
        stats['viewer_cleanup'] = cleanup
        (Path(directory)/'last_playback.json').write_text(json.dumps(stats, indent=2))
        print('[replay-complete]', json.dumps(stats), flush=True)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object', choices=['original', 'tennis_ball', 'baseball'], default='original')
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--load', type=Path, help='Replay a complete cached directory, skipping precomputation')
    parser.add_argument('--headless', action='store_true', help='Only precompute/validate; do not open a window')
    parser.add_argument('--once', action='store_true', help='Exit after playing all episodes once')
    parser.add_argument('--speed', type=float, default=1.)
    parser.add_argument('--fps', type=float, default=60.)
    parser.add_argument('--max-playback-seconds', type=float)
    args = parser.parse_args()
    if args.episodes < 1 or args.seed < 0:
        parser.error('episodes must be positive and seed nonnegative')
    if not np.isfinite(args.speed) or args.speed <= 0 or not np.isfinite(args.fps) or args.fps <= 0:
        parser.error('speed and fps must be positive finite values')
    if args.max_playback_seconds is not None and (not np.isfinite(args.max_playback_seconds) or args.max_playback_seconds <= 0):
        parser.error('max-playback-seconds must be positive and finite')
    if args.load:
        directory = args.load.resolve()
        load_recording(directory)
    else:
        checkpoint = args.checkpoint.resolve()
        config = args.config.resolve() if args.config else checkpoint.parent.parent/'resolved_config.yaml'
        tag = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
        directory = args.output.resolve() if args.output else ROOT.parent/'analysis_artifacts'/('bounce_replay_'+args.object+'_'+tag)
        precompute(checkpoint, config, args.object, args.episodes, args.seed, directory)
    if not args.headless:
        playback(directory, speed=args.speed, fps=args.fps, once=args.once,
                 max_playback_seconds=args.max_playback_seconds)
    else:
        # Loading every saved frame also checks the non-OpenGL restore path.
        model, traces, _ = load_recording(directory)
        data = mujoco.MjData(model)
        for trace in traces:
            for index in range(len(trace['time'])):
                restore_frame(model, data, trace, index)
        print('[headless-validation] all recorded qpos/qvel frames restored exactly', flush=True)


if __name__ == '__main__':
    main()
