"""python -m distillation_aux_spatial.run {check,camera-check,teacher-eval,train,eval}."""
import argparse
import copy
import importlib
import json
import random
import struct
import time
import zlib
from pathlib import Path
import numpy as np
import torch
from distillation.run import atomic_save, save_json, seed_all, wilson
from distillation.teachers import ROOT, sha256
from .core import SCHEMA, Student, EpisodeReplay, dimensions, proprioception, sequence_loss, camera_specs, auxiliary_losses
from .environments import Pool, terminal_success

TASK_IDS = {'throw': 0, 'roll': 1, 'bounce': 2}  # Scheduling/seeds only, never a network input.


def legacy(stage):
    return importlib.import_module('distillation_track.run' if stage == 'track' else 'distillation.run')


def provenance():
    hashes = {}
    for directory in ('distillation_aux_spatial', 'visual_module_experiments', 'distillation_visual_aux', 'model_experiments', 'distillation', 'distillation_track', 'gym_dcmm', 'configs', 'assets'):
        for path in (ROOT / directory).rglob('*'):
            if (path.is_file() and path.suffix in ('.py', '.xml', '.stl', '.obj', '.png', '.yaml', '.json')
                    and 'runs' not in path.parts and 'tests' not in path.parts and not path.name.startswith('tmp')):
                hashes[str(path.relative_to(ROOT))] = sha256(path)
    return hashes


def env_config(config, task):
    cfg = copy.deepcopy(config['tasks'][task])
    cfg['env_kwargs'] = dict(config['env_kwargs'], **cfg.get('env_kwargs', {}))
    cfg.update(stage=config['stage'], vision=config['vision'])
    cfg['extra_labels'] = config.get('experiment', {}).get('variant') in ('belief', 'spatial')
    return cfg


def make_student(config):
    return Student(config['stage'], config['vision'], config['model'], config['auxiliary']).to(config['device'])


def student_step(student, packet, previous, hidden, stage):
    device = student.mean.device
    robot = proprioception(packet['teacher_obs'], stage)
    with torch.no_grad():
        action, hidden, _ = student(torch.as_tensor(packet['depth'], device=device)[:, None],
            torch.as_tensor(robot, device=device)[:, None],
            torch.as_tensor(previous, device=device)[:, None], hidden)
    return action[:, 0].cpu().numpy(), hidden, robot


class Collector:
    def __init__(self, config, student, teachers, adapters, replay, start=0):
        self.config, self.student = config, student
        self.teachers, self.adapters, self.replay = teachers, adapters, replay
        self.pools, self.previous, self.hidden, self.pending = {}, {}, {}, {}
        self.state_rows = {t: [] for t in config['tasks']}
        self.collect_calibration = not student.normalizer_fitted.item()
        try:
            for task in config['tasks']:
                n = config['num_envs_per_task']
                self.pools[task] = Pool(task, env_config(config, task), n,
                                       config['seed'] + TASK_IDS[task]*10000 + start*1000000)
                self.previous[task] = np.zeros((n, student.action_dim), np.float32)
                self.hidden[task] = student.initial_state(n)
                self.pending[task] = [[] for _ in range(n)]
        except BaseException:
            self.close()
            raise

    def collect(self, steps):
        self.student.eval()
        original = legacy(self.config['stage'])
        started = time.monotonic()
        for step in range(steps):
            for task, pool in self.pools.items():
                packet = pool.obs
                action, hidden, robot = student_step(self.student, packet, self.previous[task],
                                                      self.hidden[task], self.config['stage'])
                adapter = self.adapters[task]
                target = adapter.canonical(self.teachers[task].predict(original.observation_array(packet['teacher_obs'])))
                command = adapter.executable(action)
                for i, pending in enumerate(self.pending[task]):
                    pending.append({'depth': packet['depth'][i].copy(), 'proprio': robot[i].copy(),
                                    'previous': self.previous[task][i].copy(), 'target': target[i].copy(),
                                    'object_target': packet['object_target'][i].copy(),
                                    'object_valid': packet['object_valid'][i].copy(),
                                    'world_position': packet['world_position'][i].copy(),
                                    'uv': packet['uv'][i].copy(), 'visible': packet['visible'][i].copy(),
                                    'contact_after': np.float32(0)})
                if self.collect_calibration:
                    self.state_rows[task].append(robot.copy())
                _, _, done, infos = pool.step(adapter.action_dict(command))
                for i, pending in enumerate(self.pending[task]):
                    pending[-1]['contact_after'] = np.float32(infos[i]['module_hand_contact'])
                self.previous[task] = command.copy()
                self.previous[task][done] = 0
                hidden[:, torch.as_tensor(done, device=hidden.device)] = 0
                self.hidden[task] = hidden
                for i, pending in enumerate(self.pending[task]):
                    if len(pending) > self.config['max_episode_steps']:
                        raise RuntimeError('Episode exceeds max_episode_steps; inspect termination instead of silently truncating it')
                    if done[i]:
                        self.replay.add(task, pending)
                        self.pending[task][i] = []
            if steps >= 16 and (step+1) % 16 == 0:
                print('[collect] {}/{} steps/environment elapsed={:.1f}s'.format(
                    step+1, steps, time.monotonic()-started), flush=True)

    @torch.no_grad()
    def refresh_hidden(self):
        # Weights/normalizer changed: recompute each live episode prefix with CURRENT
        # weights, rather than carrying stale hidden states across optimizer updates.
        self.student.eval()
        device = self.student.mean.device
        chunk = self.config['bptt_steps']
        for task, episodes in self.pending.items():
            for i, rows in enumerate(episodes):
                h = self.student.initial_state(1)
                for begin in range(0, len(rows), chunk):
                    part = rows[begin:begin+chunk]
                    values = [torch.as_tensor(np.stack([r[k] for r in part]), device=device)[None]
                              for k in ('depth', 'proprio', 'previous')]
                    _, h, _ = self.student(*values, h)
                self.hidden[task][:, i:i+1] = h

    def close(self):
        for pool in self.pools.values():
            pool.close()


def optimize(student, optimizer, batch, config):
    student.train()
    optimizer.zero_grad(set_to_none=True)
    hidden = None
    values = dict(loss=0., imitation_loss=0., position_loss=0., velocity_loss=0., module_loss=0.)
    lengths = batch['valid'].sum(1)
    for begin in range(0, batch['depth'].shape[1], config['bptt_steps']):
        end = begin + config['bptt_steps']
        _, hidden, logits, prediction = student.forward_aux(batch['depth'][:, begin:end], batch['proprio'][:, begin:end],
                                    batch['previous'][:, begin:end], hidden)
        loss = sequence_loss(logits, batch['target'][:, begin:end], batch['valid'][:, begin:end],
                             config['loss'], lengths)
        imitation = loss
        aux = auxiliary_losses(prediction, batch['object_target'][:, begin:end],
                               batch['object_valid'][:, begin:end], batch['valid'][:, begin:end],
                               config['auxiliary'], lengths)
        loss = imitation + config['auxiliary']['position_weight'] * aux['position']
        loss = loss + config['auxiliary']['velocity_weight'] * aux['velocity']
        module_loss = student.extra_loss(batch, begin, end, lengths)
        loss = loss + module_loss
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite imitation loss')
        loss.backward()
        hidden = hidden.detach()  # Truncated BPTT; forward memory spans the whole episode.
        for key, value in [('loss', loss), ('imitation_loss', imitation), ('position_loss', aux['position']), ('velocity_loss', aux['velocity']), ('module_loss', module_loss)]:
            values[key] += value.item()
    norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.)
    if not torch.isfinite(norm):
        raise FloatingPointError('Non-finite student gradients')
    optimizer.step()
    return values


@torch.no_grad()
def evaluate(config, adapters, student=None, teachers=None, episodes=None, viewer=False, output_file=None):
    original = legacy(config['stage'])
    metrics = importlib.import_module(original.__package__ + '.metrics')
    count = episodes or config['eval_episodes']
    results = {}
    if student is not None:
        student.eval()
    for task, adapter in adapters.items():
        print('[eval] {} {} episodes starting'.format(task, count), flush=True)
        evaluation_env = env_config(config, task)
        evaluation_env['extra_labels'] = False
        pool = Pool(task, evaluation_env, 1,
                    config['seed'] + 1000000 + TASK_IDS[task]*10000, viewer)
        previous = np.zeros((1, dimensions(config['stage'])[1]), np.float32)
        hidden = None
        rows, reward_sum, length = [], 0., 0
        try:
            for _ in range(config['max_eval_steps']):
                if student is None:
                    command = adapter.canonical(teachers[task].predict(original.observation_array(pool.obs['teacher_obs'])))
                else:
                    command, hidden, _ = student_step(student, pool.obs, previous, hidden, config['stage'])
                command = adapter.executable(command)
                _, reward, done, infos = pool.step(adapter.action_dict(command))
                previous = command.copy()
                reward_sum += float(reward[0])
                length += 1
                if length > config['max_episode_steps']:
                    raise RuntimeError('Evaluation episode exceeded max_episode_steps')
                if done[0]:
                    info = infos[0]
                    metric = config['tasks'][task]['success_metric']
                    rows.append({'success': terminal_success(info, metric), 'reward': reward_sum,
                                 'length': length, 'legacy_truncated': bool(info['_truncated']),
                                 'reason': str(info.get('terminated_reason', 'unspecified')),
                                 'diagnostics': metrics.terminal_diagnostics(info)})
                    results[task] = metrics.summarize(rows, metric, count)
                    results[task].update(policy_role='teacher' if student is None else 'visual_student',
                        success_definition=config['tasks'][task].get('success_definition'),
                        evaluation_revision=config.get('evaluation_revision'),
                        seed=config['seed'], stage=config['stage'], camera=config['vision'],
                        wilson_95=wilson(sum(r['success'] for r in rows), len(rows)))
                    if output_file:
                        save_json(output_file, results)
                    if len(rows) == 1 or len(rows) % 10 == 0 or len(rows) == count:
                        print('[eval] {} {}/{} successes={}'.format(task, len(rows), count,
                              results[task]['successes']), flush=True)
                    previous.fill(0)
                    hidden = None
                    reward_sum, length = 0., 0
                    if len(rows) == count:
                        break
            if len(rows) != count:
                raise RuntimeError('Evaluation budget exhausted')
        finally:
            pool.close()
    return results


def package(student, config, metadata, iteration):
    return {'schema': SCHEMA, 'student': student.state_dict(), 'config': config,
            'metadata': metadata, 'iteration': iteration}


def rng_state():
    n = np.random.get_state()
    return {'torch': torch.get_rng_state(), 'python': random.getstate(),
            'numpy': (n[0], n[1].tolist(), *n[2:]),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state, saved_device=None, device=None):
    torch.set_rng_state(state['torch'])
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
    from model_experiments.visual_resume import restore_cuda_rng
    restore_cuda_rng(state['cuda'], saved_device, device)


def train(config, teachers, adapters, output, resume=None, allow_source_changes=False):
    student = make_student(config)
    optimizer = torch.optim.Adam(student.parameters(), lr=config['learning_rate'])
    replay = EpisodeReplay(config['tasks'], config['capacity_frames_per_task'])
    metadata = {'teacher_hashes': {t: teacher.hash for t, teacher in teachers.items()},
                'source_asset_sha256': provenance()}
    start, best, baseline = 0, -float('inf'), None
    if resume:
        saved = torch.load(resume, map_location='cpu', weights_only=True)
        from model_experiments.visual_resume import validate_resume
        changes, reset_best = validate_resume(saved, config, metadata, SCHEMA, allow_source_changes)
        save_json(output / 'resume_audit.json', {'checkpoint': str(resume),
            'changed_sources': changes, 'old_config': saved['config'], 'new_config': config,
            'source_changes_accepted': allow_source_changes})
        if changes:
            print('[resume] accepted source changes: ' + ', '.join(changes), flush=True)
        student.load_state_dict(saved['student'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        replay.load_state_dict(saved['replay'])
        start, best, baseline = saved['iteration'], saved['best_score'], saved['baseline']
        if reset_best:
            best = -float('inf')
        restore_rng(saved['rng'], saved['config']['device'], config['device'])
        print('Resuming model/optimizer/replay; simulator episodes restart.', flush=True)
    if start >= config['iterations']:
        raise ValueError('iterations must exceed the resumed iteration')
    if config.get('skip_teacher_baseline'):
        baseline = {task: {'successes': 0, 'episodes': 0, 'success_rate': 0.0}
                    for task in config['tasks']}
        print('[baseline] teacher evaluation skipped; best uses minimum student task success', flush=True)
    if baseline is None:
        baseline = evaluate(config, adapters, teachers=teachers, episodes=config['baseline_episodes'],
                            output_file=output / 'teacher_baseline.json')
    if not config.get('skip_teacher_baseline') and any(r['successes'] == 0 for r in baseline.values()) and not config.get('allow_zero_teacher_success'):
        raise RuntimeError('Zero teacher success: inspect baseline. For smoke tests only use --allow-zero-teacher-success')
    collector = Collector(config, student, teachers, adapters, replay, start)
    try:
        if not student.normalizer_fitted.item():
            print('[calibration] collecting robot-only normalization statistics', flush=True)
            collector.collect(config['calibration_steps'])
            states = [np.concatenate(rows) for rows in collector.state_rows.values()]
            student.fit_normalizer(np.concatenate(states))
            collector.collect_calibration = False
            collector.state_rows.clear()
            collector.refresh_hidden()
        if not replay.ready():
            print('[warmup] waiting for at least one completed episode per task', flush=True)
            for _ in range(config['max_episode_steps'] + 1):
                collector.collect(1)
                if replay.ready():
                    break
            if not replay.ready():
                raise RuntimeError('No complete episodes for replay')
        for iteration in range(start + 1, config['iterations'] + 1):
            before = time.monotonic()
            collector.collect(config['rollout_steps'])
            losses = [optimize(student, optimizer,
                       replay.sample(config['batch_episodes_per_task'], config['device']), config)
                      for _ in range(config['updates_per_iteration'])]
            collector.refresh_hidden()
            row = {'iteration': iteration, **{key: float(np.mean([v[key] for v in losses])) for key in losses[0]},
                   'seconds': time.monotonic()-before, 'replay_frames': dict(replay.sizes)}
            checkpoint = package(student, config, metadata, iteration)
            atomic_save(output / 'student_last.pth', checkpoint)
            if iteration % config['eval_every'] == 0 or iteration == config['iterations']:
                saved_rng = rng_state()
                results = evaluate(config, adapters, student=student,
                                   output_file=output / ('eval_%06d.json' % iteration))
                restore_rng(saved_rng)
                score = min(results[t]['success_rate']-baseline[t]['success_rate'] for t in results)
                row['min_task_success' if config.get('skip_teacher_baseline') else 'worst_teacher_gap'] = score
                if score > best:
                    best = score
                    atomic_save(output / 'student_best.pth', checkpoint)
            if iteration % config['save_every'] == 0 or iteration == config['iterations']:
                atomic_save(output / 'resume_last.pth', dict(checkpoint, optimizer=optimizer.state_dict(),
                    replay=replay.state_dict(), best_score=best, baseline=baseline, rng=rng_state()))
            with (output / 'metrics.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True)
    finally:
        collector.close()


def write_gray_png(path, pixels):
    """Portable camera contact sheet without optional image-library dependencies."""
    def chunk(kind, payload):
        return struct.pack('!I', len(payload)) + kind + payload + struct.pack('!I', zlib.crc32(kind+payload) & 0xffffffff)
    h, w = pixels.shape
    header = struct.pack('!2I5B', w, h, 8, 0, 0, 0, 0)
    data = b''.join(b'\x00' + row.tobytes() for row in pixels)
    Path(path).write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', header) +
                           chunk(b'IDAT', zlib.compress(data)) + chunk(b'IEND', b''))


def camera_check(config, teachers, adapters, output, steps):
    # Save real depth streams along a teacher rollout, so users can inspect visibility
    # before paying for a long training run. This is not a success-rate evaluation.
    original = legacy(config['stage'])
    report = {}
    for task, adapter in adapters.items():
        evaluation_env = env_config(config, task)
        evaluation_env['extra_labels'] = False
        pool = Pool(task, evaluation_env, 1, config['seed'] + TASK_IDS[task]*10000)
        frames, ends = [], []
        try:
            for _ in range(steps):
                frames.append(pool.obs['depth'][0].copy())
                command = adapter.canonical(teachers[task].predict(original.observation_array(pool.obs['teacher_obs'])))
                _, _, done, _ = pool.step(adapter.action_dict(command))
                ends.append(bool(done[0]))
        finally:
            pool.close()
        array = np.stack(frames)
        specs = camera_specs(config['vision'])
        np.savez_compressed(output / (task + '_depth.npz'), depth_normalized=array,
                            done=np.asarray(ends), camera_names=np.array([s['name'] for s in specs]),
                            near=np.array([s['near'] for s in specs]), far=np.array([s['far'] for s in specs]))
        # Ordinary PNG contact sheet; no matplotlib/PIL dependency in training.
        indices = np.linspace(0, len(array)-1, min(12, len(array)), dtype=int)
        tiles = [np.concatenate([(255*(1-view.astype(np.float32))).astype(np.uint8)
                                 for view in array[i]], axis=1) for i in indices]
        columns = 4
        while len(tiles) % columns:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.concatenate([np.concatenate(tiles[i:i+columns], axis=1)
                                for i in range(0, len(tiles), columns)], axis=0)
        write_gray_png(output / (task + '_depth.png'), sheet)
        report[task] = {'frames': len(array), 'shape': list(array.shape),
                        'cameras_left_to_right': specs,
                        'min_normalized': float(array.min()), 'max_normalized': float(array.max()),
                        'frame_std_mean': float(array.astype(np.float32).std(axis=(-2, -1)).mean()),
                        'note': 'Inspect contact sheet/sequence for ball visibility; statistics alone do not prove visibility.'}
    save_json(output / 'camera_report.json', report)
    print(json.dumps(report, indent=2), flush=True)


def validate(config):
    dimensions(config['stage'])
    for k in ('iterations', 'num_envs_per_task', 'rollout_steps', 'updates_per_iteration',
              'batch_episodes_per_task', 'capacity_frames_per_task', 'calibration_steps', 'eval_every',
              'save_every', 'eval_episodes', 'baseline_episodes', 'max_eval_steps', 'max_episode_steps', 'bptt_steps'):
        if type(config[k]) is not int or config[k] < 1:
            raise ValueError(k + ' must be a positive integer')
    if config['capacity_frames_per_task'] < config['max_episode_steps']:
        raise ValueError('Replay capacity must fit one full episode')
    if config['loss'] not in ('bce', 'mse'):
        raise ValueError('loss must be bce or mse')
    if not config['tasks']:
        raise ValueError('No tasks selected')
    # A label-free deployment must not secretly select task-specific action scales.
    original = legacy(config['stage'])
    if config.get('experiment', {}).get('variant') != 'spatial':
        raise ValueError('Config belongs to a different module experiment')
    adapters = original.make_adapters(config)
    reference = next(iter(adapters.values()))
    for adapter in adapters.values():
        if not np.array_equal(adapter.scale, reference.scale) or not np.all(adapter.mask == 1):
            raise ValueError('Label-free control requires the same physical scales and full action mapping for all tasks')
    v, m = config['vision'], config['model']
    camera_specs(v)
    if not 0 < v['near'] < v['far'] or min(v['height'], v['width']) < 48:
        raise ValueError('Invalid depth range/image size')
    if not (len(m['cnn_channels']) == len(m['cnn_kernels']) == len(m['cnn_strides']) == 3):
        raise ValueError('Expected three CNN layers')
    if v['gl_backend'] not in ('auto', 'egl', 'glfw', 'osmesa'):
        raise ValueError('Unsupported GL backend')
    aux = config['auxiliary']
    if (len(aux['scales']) != 6 or not np.isfinite(aux['scales']).all()
            or min(aux['scales']) <= 0):
        raise ValueError('Auxiliary scales must be six positive finite values')
    for key in ('position_weight', 'velocity_weight'):
        if not np.isfinite(aux[key]) or aux[key] < 0:
            raise ValueError('Auxiliary weights must be finite and nonnegative')
    return adapters


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['check', 'camera-check', 'teacher-eval', 'train', 'eval'])
    p.add_argument('--stage', choices=['track', 'catch'], default='catch')
    p.add_argument('--config')
    p.add_argument('--tasks', nargs='+', choices=list(TASK_IDS))
    p.add_argument('--device')
    p.add_argument('--output', required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--resume')
    p.add_argument('--allow-source-changes', action='store_true',
                   help='Accept reviewed source changes on resume; records every changed path')
    p.add_argument('--image-size', type=int, help='Square camera resolution; training from scratch only')
    p.add_argument('--iterations', type=int)
    p.add_argument('--num-envs', type=int)
    p.add_argument('--episodes', type=int)
    p.add_argument('--baseline-episodes', type=int)
    p.add_argument('--eval-every', type=int)
    p.add_argument('--camera-steps', type=int, default=80)
    p.add_argument('--seed', type=int)
    p.add_argument('--gl-backend', choices=['auto', 'egl', 'glfw', 'osmesa'])
    p.add_argument('--egl-device-id', type=int, help='Explicit MuJoCo EGL device index (not necessarily CUDA logical index)')
    p.add_argument('--viewer', action='store_true')
    p.add_argument('--allow-zero-teacher-success', action='store_true')
    p.add_argument('--skip-teacher-baseline', action='store_true',
                   help='Skip initial teacher evaluation; select best by minimum student task success')
    args = p.parse_args()
    if args.egl_device_id is not None:
        if args.egl_device_id < 0:
            p.error('--egl-device-id must be nonnegative')
        import os
        os.environ['MUJOCO_EGL_DEVICE_ID'] = str(args.egl_device_id)
    saved = None
    if args.command == 'eval' or args.resume:
        path = args.checkpoint if args.command == 'eval' else args.resume
        if not path:
            p.error('eval requires --checkpoint')
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved.get('schema') != SCHEMA:
            raise ValueError('Expected auxiliary student checkpoint; baseline checkpoints cannot resume this experiment')
        config = copy.deepcopy(saved['config'])
    else:
        path = Path(args.config) if args.config else ROOT / 'distillation_aux_spatial/configs' / (args.stage + '.json')
        config = json.loads(path.read_text(encoding='utf-8-sig'))
    if args.image_size is not None:
        if args.image_size < 16 or args.resume or args.command == 'eval':
            p.error('--image-size requires a fresh run and size >= 16')
        config['vision']['height'] = config['vision']['width'] = args.image_size
    if args.resume and args.command != 'train':
        p.error('--resume is train only')
    if args.viewer and args.command not in ('eval', 'teacher-eval'):
        p.error('--viewer is evaluation only')
    if args.tasks:
        config['tasks'] = {t: config['tasks'][t] for t in args.tasks}
    for flag, key in [('iterations', 'iterations'), ('num_envs', 'num_envs_per_task'),
                      ('baseline_episodes', 'baseline_episodes'), ('eval_every', 'eval_every'), ('seed', 'seed')]:
        if getattr(args, flag) is not None:
            config[key] = getattr(args, flag)
    if args.episodes is not None:
        if args.episodes < 1:
            p.error('--episodes must be positive')
        config['eval_episodes'] = args.episodes
    if args.camera_steps < 1:
        p.error('--camera-steps must be positive')
    if args.allow_zero_teacher_success:
        config['allow_zero_teacher_success'] = True
    if args.skip_teacher_baseline:
        config['skip_teacher_baseline'] = True
    if args.gl_backend:
        config['vision']['gl_backend'] = args.gl_backend
    config['device'] = args.device or ('cpu' if args.command == 'check' else config['device'])
    if config['device'].startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu for local checks')
    import os
    print('[devices] torch={} EGL_DEVICE={} CUDA_VISIBLE_DEVICES={}'.format(
        config['device'], os.environ.get('MUJOCO_EGL_DEVICE_ID', 'default'),
        os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')), flush=True)
    adapters = validate(config)
    torch.set_num_threads(1)
    seed_all(config['seed'])
    output = Path(args.output)
    if args.command == 'train' and not args.resume and (output / 'resolved_config.json').exists():
        raise FileExistsError('Run exists; choose a new --output or --resume resume_last.pth')
    output.mkdir(parents=True, exist_ok=True)
    if args.command == 'eval':
        if saved['metadata']['source_asset_sha256'] != provenance():
            raise ValueError('Source/assets differ from checkpoint; review differences before evaluation')
        student = make_student(config)
        student.load_state_dict(saved['student'], strict=True)
        save_json(output / 'evaluation_config.json', dict(config=config, iteration=saved['iteration'],
                                                         checkpoint_sha256=sha256(args.checkpoint)))
        evaluate(config, adapters, student=student, viewer=args.viewer,
                 output_file=output / 'student_evaluation.json')
        return
    original = legacy(config['stage'])
    teachers = original.make_teachers(config)
    report = original.check(config, teachers)
    print(json.dumps(report, indent=2), flush=True)
    save_json(output / 'teacher_check.json', report)
    if not all(r['interface_ready'] for r in report.values()):
        raise ValueError('Teacher interface verification failed')
    if args.command == 'check':
        student = make_student(config)
        print('Visual student: {} proprio + {} previous + depth {} cameras={} -> independent CNNs -> GRU -> {} actions; no task/object input'.format(
            student.proprio_dim, student.action_dim, student.image_shape, student.camera_names, student.action_dim), flush=True)
    elif args.command == 'camera-check':
        camera_check(config, teachers, adapters, output, args.camera_steps)
    elif args.command == 'teacher-eval':
        evaluate(config, adapters, teachers=teachers, viewer=args.viewer,
                 output_file=output / 'teacher_baseline.json')
    else:
        save_json(output / 'resolved_config.json', config)
        train(config, teachers, adapters, output, args.resume, args.allow_source_changes)


if __name__ == '__main__':
    main()
