"""Run from catch_it_copy: python -m distillation.run --help."""
import argparse
import copy
import json
import random
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from .core import (ActionAdapter, Replay, SCHEMA, TASK_IDS, Student,
                   imitation_loss, observation_array, student_input)
from .teachers import ROOT, Teacher, original_method, sha256


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def atomic_save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def provenance():
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = 'unknown'
    sources = {}
    for folder in ['distillation', 'gym_dcmm', 'configs/env']:
        for path in (ROOT / folder).rglob('*.py'):
            sources[str(path.relative_to(ROOT))] = sha256(path)
    for path in (ROOT / 'urdf').rglob('*'):
        if path.is_file():
            sources[str(path.relative_to(ROOT))] = sha256(path)
    return {'git_revision': revision, 'source_asset_sha256': sources}


def make_teachers(config):
    return {task: Teacher(cfg, config['device']) for task, cfg in config['tasks'].items()}


def make_adapters(config):
    return {task: ActionAdapter(cfg) for task, cfg in config['tasks'].items()}


def check(config, teachers):
    from types import SimpleNamespace
    result = {}
    raw = np.random.default_rng(13).normal(size=(32, 30)).astype(np.float32)
    obs = {'base': {'v_lin_2d': raw[:, :2]},
           'arm': {'ee_pos3d': raw[:, 2:5], 'ee_quat': raw[:, 5:9], 'ee_v_lin_3d': raw[:, 9:12]},
           'object': {'pos3d': raw[:, 12:15], 'v_lin_3d': raw[:, 15:18]}, 'hand': raw[:, 18:]}
    for task, teacher in teachers.items():
        path = teacher.family / 'ppo_dcmm_catch_two_stage.py'
        reference_obs = original_method(path, 'obs2tensor')(SimpleNamespace(device=teacher.device), obs)
        np.testing.assert_allclose(reference_obs.cpu().numpy(), observation_array(obs), atol=1e-7)
        row = {'sha256': teacher.hash, 'native_action_dim': teacher.action_dim,
               'inference_max_error': teacher.parity_check(raw), 'observation_parity': True}
        try:
            adapter = ActionAdapter(config['tasks'][task])
            native = teacher.predict(raw)
            mapped = adapter.canonical(native)
            if teacher.action_dim == 20 and adapter.indices == list(range(20)):
                proxy = SimpleNamespace(env=SimpleNamespace(call=lambda _: ['Catching']),
                    action_catch_denorm=config['tasks'][task]['action_scales'])
                reference = original_method(path, 'action2dict')(proxy, torch.from_numpy(native))
                for key, value in adapter.action_dict(mapped).items():
                    np.testing.assert_allclose(value, reference[key], atol=1e-7)
                row['action_parity'] = True
            else:
                row['action_parity'] = 'custom mapping requires original execution verification'
            row['interface_ready'] = True
        except ValueError as exc:
            row.update(interface_ready=False, reason=str(exc))
        row['closed_loop_verified'] = False
        result[task] = row
    return result


def env_config(config, task):
    result = copy.deepcopy(config['tasks'][task])
    result['env_kwargs'] = dict(config['env_kwargs'], **result.get('env_kwargs', {}))
    return result


def wilson(successes, count):
    z, p = 1.96, successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * ((p * (1-p) / count + z*z / (4*count*count)) ** 0.5) / denominator
    return [max(0., center-radius), min(1., center+radius)]


@torch.no_grad()
def evaluate(config, adapters, student=None, teachers=None, episodes=None, viewer=False):
    from .environments import Pool, terminal_success
    episodes = episodes or config['eval_episodes']
    results = {}
    if student is not None:
        student.eval()
    for task, adapter in adapters.items():
        pool = Pool(task, env_config(config, task), 1,
                    config['seed'] + 1000000 + TASK_IDS[task] * 10000, viewer)
        previous = np.zeros((1, 20), np.float32)
        rows, total_reward, length = [], 0., 0
        try:
            for _ in range(config['max_eval_steps']):
                if student is None:
                    command = adapter.canonical(teachers[task].predict(observation_array(pool.obs)))
                else:
                    x = student_input(pool.obs, previous, task)
                    command = student(torch.as_tensor(x, device=config['device'])).cpu().numpy()
                command = adapter.executable(command)
                _, reward, done, infos = pool.step(adapter.action_dict(command))
                previous = command.copy()
                total_reward += float(reward[0])
                length += 1
                if done[0]:
                    info = infos[0]
                    rows.append({'success': terminal_success(info, config['tasks'][task]['success_metric']),
                                 'environment_success': bool(info.get('success', False)),
                                 'legacy_truncated': bool(info['_truncated']),
                                 'reward': total_reward, 'length': length,
                                 'reason': str(info.get('terminated_reason', 'unspecified'))})
                    total_reward, length = 0., 0
                    previous.fill(0)
                    if len(rows) == episodes:
                        break
            if len(rows) != episodes:
                raise RuntimeError('Evaluation step budget exhausted before requested episodes completed')
        finally:
            pool.close()
        n = sum(r['success'] for r in rows)
        results[task] = {'success_metric': config['tasks'][task]['success_metric'],
                         'successes': n, 'episodes': episodes, 'success_rate': n / episodes,
                         'wilson_95': wilson(n, episodes),
                         'legacy_truncated_rate': np.mean([r['legacy_truncated'] for r in rows]).item(),
                         'mean_reward': np.mean([r['reward'] for r in rows]).item(), 'records': rows}
        print(task, 'evaluation:', n, '/', episodes, flush=True)
    return results


def deployment(student, config, metadata, iteration):
    return {'schema': SCHEMA, 'student': student.state_dict(), 'config': config,
            'metadata': metadata, 'iteration': iteration, 'task_ids': TASK_IDS}


def train(config, teachers, adapters, output, resume=None):
    from .environments import Pool
    device = config['device']
    student = Student(config['hidden']).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=config['learning_rate'])
    replay = Replay(config['tasks'], config['capacity_per_task'])
    metadata = provenance()
    metadata['teacher_hashes'] = {t: teacher.hash for t, teacher in teachers.items()}
    start, best, baseline = 0, -float('inf'), None
    if resume:
        saved = torch.load(resume, map_location='cpu', weights_only=True)
        expected, actual = copy.deepcopy(saved['config']), copy.deepcopy(config)
        # Permit extending the iteration budget, but keep data/control contracts fixed.
        expected.pop('iterations'); actual.pop('iterations')
        if saved['schema'] != SCHEMA or expected != actual:
            raise ValueError('Resume configuration/schema differs (only iterations may change)')
        if saved['metadata']['teacher_hashes'] != metadata['teacher_hashes']:
            raise ValueError('Resume teachers changed')
        if saved['metadata']['source_asset_sha256'] != metadata['source_asset_sha256']:
            raise ValueError('Resume source/assets changed; use a new run after reviewing changes')
        student.load_state_dict(saved['student'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        replay.load_state_dict(saved['replay'])
        start, best, baseline = saved['iteration'], saved['best_score'], saved['baseline']
        torch.set_rng_state(saved['torch_rng'])
        if saved.get('cuda_rng') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        state = saved['numpy_rng']
        np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), *state[2:]))
        random.setstate(saved['python_rng'])
        print('Resuming optimizer/replay/RNG; simulator episodes restart (not bitwise continuation).', flush=True)
    if baseline is None:
        baseline = evaluate(config, adapters, teachers=teachers, episodes=config['baseline_episodes'])
        save_json(output / 'teacher_baseline.json', baseline)
        if any(row['successes'] == 0 for row in baseline.values()):
            raise RuntimeError('At least one teacher has zero measured success; inspect baseline before distillation')
    pools, previous = {}, {}
    try:
        for task in config['tasks']:
            pools[task] = Pool(task, env_config(config, task), config['num_envs_per_task'],
                               config['seed'] + TASK_IDS[task] * 10000 + start * 1000000)
            previous[task] = np.zeros((config['num_envs_per_task'], 20), np.float32)

        def collect(steps, beta):
            student.eval()
            for _ in range(steps):
                for task, pool in pools.items():
                    adapter = adapters[task]
                    x = student_input(pool.obs, previous[task], task)
                    label = adapter.canonical(teachers[task].predict(observation_array(pool.obs)))
                    with torch.no_grad():
                        prediction = student(torch.as_tensor(x, device=device)).cpu().numpy()
                    replay.add(task, x, label, adapter.mask)
                    select = np.random.random((len(x), 1)) < beta
                    command = adapter.executable(np.where(select, label, prediction))
                    _, _, done, _ = pool.step(adapter.action_dict(command))
                    previous[task] = command.copy()
                    previous[task][done] = 0

        if not student.normalizer_fitted.item():
            # Online student rollout only; no offline BC and no teacher execution by default.
            collect(config['calibration_steps'], config['beta_start'])
            states = torch.cat([d['x'][:d['size']] for d in replay.data.values()]).to(device)
            student.fit_normalizer(states)
        for iteration in range(start + 1, config['iterations'] + 1):
            before = time.monotonic()
            beta = config['beta_start'] * max(0., 1 - (iteration-1) / config['beta_decay_iterations'])
            collect(config['rollout_steps'], beta)
            student.train()
            losses = []
            for _ in range(config['updates_per_iteration']):
                x, target, mask = replay.sample(config['batch_per_task'], device)
                loss = imitation_loss(student(x), target, mask)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite imitation loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.)
                optimizer.step()
                losses.append(loss.item())
            row = {'iteration': iteration, 'beta': beta, 'loss': float(np.mean(losses)),
                   'seconds': time.monotonic()-before,
                   'replay_sizes': {t: d['size'] for t, d in replay.data.items()}}
            package = deployment(student, config, metadata, iteration)
            if iteration % config['eval_every'] == 0 or iteration == config['iterations']:
                # Evaluation must not advance the training RNG stream.
                rng = (torch.get_rng_state(), np.random.get_state(), random.getstate())
                results = evaluate(config, adapters, student=student)
                torch.set_rng_state(rng[0]); np.random.set_state(rng[1]); random.setstate(rng[2])
                save_json(output / ('eval_%06d.json' % iteration), results)
                # Maximize worst task difference from its teacher, not average reward.
                score = min(results[t]['success_rate'] - baseline[t]['success_rate'] for t in results)
                row['worst_teacher_gap'] = score
                if score > best:
                    best = score
                    atomic_save(output / 'student_best.pth', package)
            atomic_save(output / 'student_last.pth', package)
            if iteration % config['eval_every'] == 0 or iteration == config['iterations']:
                np_state = np.random.get_state()
                full = dict(package, optimizer=optimizer.state_dict(), replay=replay.state_dict(),
                    best_score=best, baseline=baseline, torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    numpy_rng=(np_state[0], np_state[1].tolist(), *np_state[2:]),
                    python_rng=random.getstate())
                atomic_save(output / 'resume_last.pth', full)
            with (output / 'metrics.jsonl').open('a', encoding='utf-8') as log:
                log.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True)
    finally:
        for pool in pools.values():
            pool.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'teacher-eval', 'train', 'eval'])
    parser.add_argument('--config', default=str(ROOT / 'distillation/configs/three_tasks.json'))
    parser.add_argument('--tasks', nargs='+', choices=list(TASK_IDS))
    parser.add_argument('--device')
    parser.add_argument('--output', default='distillation/runs/three_tasks')
    parser.add_argument('--checkpoint', help='Student-only deployment checkpoint for eval')
    parser.add_argument('--resume', help='Full resume_last.pth for train')
    parser.add_argument('--episodes', type=int)
    parser.add_argument('--iterations', type=int)
    parser.add_argument('--num-envs', type=int)
    parser.add_argument('--viewer', action='store_true', help='Evaluation only')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    package = None
    if args.command == 'eval':
        if not args.checkpoint:
            parser.error('eval requires --checkpoint')
        package = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        if package['schema'] != SCHEMA:
            raise ValueError('Unknown student schema')
        config = copy.deepcopy(package['config'])
    if args.tasks:
        if any(t not in config['tasks'] for t in args.tasks):
            raise ValueError('Requested task was not trained/configured')
        config['tasks'] = {t: config['tasks'][t] for t in args.tasks}
    if args.device:
        config['device'] = args.device
    elif args.command == 'check':
        config['device'] = 'cpu'
    if args.iterations is not None:
        config['iterations'] = args.iterations
    if args.num_envs is not None:
        config['num_envs_per_task'] = args.num_envs
    if args.episodes is not None and args.episodes < 1:
        parser.error('--episodes must be positive')
    for key in ['iterations', 'num_envs_per_task', 'rollout_steps', 'updates_per_iteration',
                'batch_per_task', 'capacity_per_task', 'calibration_steps', 'eval_every',
                'eval_episodes', 'baseline_episodes', 'beta_decay_iterations', 'max_eval_steps']:
        if config[key] < 1:
            raise ValueError(key + ' must be positive')
    if not 0 <= config['beta_start'] <= 1:
        raise ValueError('beta_start must be in [0,1]')
    if str(config['device']).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; pass --device cpu or use a CUDA PyTorch environment')
    if args.viewer and args.command not in ('teacher-eval', 'eval'):
        parser.error('--viewer is evaluation only')
    seed_all(config['seed'])
    torch.set_num_threads(1)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.command == 'eval':
        current = provenance()['source_asset_sha256']
        recorded = package['metadata']['source_asset_sha256']
        changed = [name for name, digest in recorded.items()
                   if not name.startswith('distillation') and current.get(name) != digest]
        if changed:
            raise ValueError('Environment/assets differ from training checkpoint: ' + ', '.join(changed[:5]))
        student = Student(config['hidden']).to(config['device'])
        student.load_state_dict(package['student'], strict=True)
        results = evaluate(config, make_adapters(config), student=student,
                           episodes=args.episodes, viewer=args.viewer)
        save_json(output / 'student_evaluation.json', results)
        return
    teachers = make_teachers(config)
    report = check(config, teachers)
    save_json(output / 'teacher_check.json', report)
    print(json.dumps(report, indent=2), flush=True)
    if args.command == 'check':
        if not all(r['interface_ready'] for r in report.values()):
            raise SystemExit(2)
        return
    adapters = make_adapters(config)  # Fail before creating any simulator if mapping unresolved.
    if args.command == 'teacher-eval':
        results = evaluate(config, adapters, teachers=teachers,
                           episodes=args.episodes, viewer=args.viewer)
        save_json(output / 'teacher_baseline.json', results)
        return
    if (output / 'student_last.pth').exists() and not args.resume:
        raise FileExistsError('Run already exists; use a new --output or --resume')
    save_json(output / 'resolved_config.json', config)
    train(config, teachers, adapters, output, args.resume)


if __name__ == '__main__':
    main()
