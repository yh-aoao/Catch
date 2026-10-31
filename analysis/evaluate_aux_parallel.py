"""Vectorized Future/Contact evaluation using audited historical sources.

--num-envs controls simultaneous environments within EACH task. --episodes is
the TOTAL completed episodes per task, distributed with fixed worker quotas.
Tasks run sequentially; training code and checkpoint weights are untouched.
"""
import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tarfile
import time


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT/'analysis_artifacts/distill_best_multiseed_20261029_v1'
TASKS = ('throw', 'roll', 'bounce')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def episode_quotas(episodes, environments):
    if episodes < 1 or environments < 1:
        raise ValueError('episodes and num-envs must be positive')
    count = min(episodes, environments)
    return [episodes//count + int(i < episodes % count) for i in range(count)]


def audit_for(variant):
    if variant == 'future':
        path, key, prefix = PRIOR/'source_audit.json', 'future', 'future/'
    else:
        path = ROOT/'analysis_artifacts/contact_best_last_20261030_v1/source_audit.json'
        key, prefix = 'best', 'belief/'
    return path, json.loads(path.read_text())[key], prefix


def restore_and_verify(audit, archive_prefix):
    """Restore only missing files, then hash every source and required asset."""
    import fcntl
    source = Path(audit['source_root'])
    expected = {item['path']: item['actual'] for item in audit['files']}
    expected.update({item['path']: item['sha256'] for item in audit.get('additional_required_assets', [])})
    for rel in expected:
        if Path(rel).is_absolute() or '..' in Path(rel).parts:
            raise ValueError('Unsafe source path: '+rel)
    source.parent.mkdir(parents=True, exist_ok=True)
    with (source.parent/(source.name+'.restore.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        missing = set()
        for rel, digest in expected.items():
            path = source/rel
            if not path.exists():
                missing.add(rel)
            elif sha(path) != digest:
                raise RuntimeError('Audited source changed: '+str(path))
        if missing:
            archive = PRIOR/'historical_sources_and_assets.tar.gz'
            print('[sources] restoring audited historical files', flush=True)
            with tarfile.open(archive, 'r|gz') as tf:
                for member in tf:
                    if not member.name.startswith(archive_prefix):
                        continue
                    rel = member.name[len(archive_prefix):]
                    if rel not in missing:
                        continue
                    if not member.isfile():
                        raise RuntimeError('Unexpected archive member: '+member.name)
                    contents = tf.extractfile(member).read()
                    if hashlib.sha256(contents).hexdigest() != expected[rel]:
                        raise RuntimeError('Historical archive hash mismatch: '+rel)
                    path = source/rel
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(contents)
                    missing.remove(rel)
            # Contact's six entry files are absent from the old Belief archive.
            # Use current copies only when their checkpoint-pinned hashes match.
            for rel in sorted(missing):
                current = ROOT/'catch_it_copy'/rel
                if not current.is_file() or sha(current) != expected[rel]:
                    raise RuntimeError('Cannot restore exact checkpoint source: '+rel)
                path = source/rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(current.read_bytes())
        for rel, digest in expected.items():
            if sha(source/rel) != digest:
                raise RuntimeError('Source verification failed: '+rel)
    return source


def write_json(path, value, metrics):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(metrics.json_value(value), indent=2, ensure_ascii=False))
    temporary.replace(path)


def initial_state(packet):
    value = json.loads(str(packet['teacher_obs']['_evaluation_initial_json']))
    value['depth_sha256'] = hashlib.sha256(packet['depth'].tobytes()).hexdigest()
    return value


def evaluate_task(run, config, student, adapter, args, task, output, protocol, metrics, stack_tree):
    import numpy as np
    import torch

    quotas = episode_quotas(args.episodes, args.num_envs)
    count = len(quotas)
    base_seed = args.seed+1000000+run.TASK_IDS[task]*10000
    task_protocol = dict(protocol, task=task, environments=count, worker_episode_quotas=quotas,
                         constructor_seeds=[base_seed+i*1009 for i in range(count)],
                         reset_seed='constructor_seed + local_episode_index*100003 (NumPy modulo 2**32)',
                         episode_id='worker_id + local_episode_index*environments')
    output.mkdir(parents=True)
    write_json(output/'protocol.json', task_protocol, metrics)
    cfg = run.env_config(config, task)
    cfg.update(extra_labels=False, env_factory='distill_eval_capture:make_env',
               worker_timeout_seconds=args.worker_timeout)
    print('[eval] {}: {} total episodes, {} parallel environments, initializing'.format(
        task, args.episodes, count), flush=True)
    start = time.monotonic()
    pool = None
    try:
        pool = run.Pool(task, cfg, count, base_seed)
        packets = [None]*count

        def unstack(tree, i):
            if isinstance(tree, dict):
                return {key: unstack(value, i) for key, value in tree.items()}
            return tree[i]

        for i in range(count):
            packets[i] = unstack(pool.obs, i)
        initials = [initial_state(packet) for packet in packets]
        previous = np.zeros((count, student.action_dim), np.float32)
        hidden = student.initial_state(count)
        rewards, lengths, completed = [0.]*count, [0]*count, [0]*count
        rows = []
        last_report, last_count = start, 0
        with (output/'episodes_partial.jsonl').open('x', buffering=1) as partial:
            while len(rows) < args.episodes:
                active = [i for i in range(count) if completed[i] < quotas[i]]
                # Reset-only capture metadata has different keys from step obs.
                # Stack just the robot fields used by the original policy.
                robot_keys = ('base', 'arm') + (('hand',) if config['stage'] == 'catch' else ())
                robot_obs = stack_tree([{key: packets[i]['teacher_obs'][key] for key in robot_keys}
                                        for i in active])
                proprio = run.proprioception(robot_obs, config['stage'])
                depth = np.stack([packets[i]['depth'] for i in active])
                with torch.inference_mode():
                    inputs = (torch.as_tensor(depth, device=args.device)[:, None],
                              torch.as_tensor(proprio, device=args.device)[:, None],
                              torch.as_tensor(previous[active], device=args.device)[:, None])
                    state = hidden[:, active].contiguous()
                    chunk = args.inference_batch_size or len(active)
                    actions, states = [], []
                    for begin in range(0, len(active), chunk):
                        end = min(begin+chunk, len(active))
                        action, new_state, _ = student(*(v[begin:end] for v in inputs),
                                                       state[:, begin:end].contiguous())
                        actions.append(action)
                        states.append(new_state)
                    command = adapter.executable(torch.cat(actions, 0)[:, 0].cpu().numpy())
                    hidden[:, active] = torch.cat(states, 1)
                if not np.isfinite(command).all():
                    raise FloatingPointError('Non-finite policy action')
                physical = adapter.action_dict(command)
                # Send every active environment before waiting for any response.
                for offset, worker in enumerate(active):
                    previous[worker] = command[offset]
                    pool.pipes[worker].send(('step', {key: value[offset] for key, value in physical.items()}))
                for worker in active:
                    packet, reward, done, info = pool._receive(pool.pipes[worker])
                    packets[worker] = packet
                    rewards[worker] += float(reward)
                    lengths[worker] += 1
                    if lengths[worker] > config['max_episode_steps']:
                        raise RuntimeError('Episode exceeded original max_episode_steps')
                    if not done:
                        continue
                    local_episode = completed[worker]
                    row = dict(seed=args.seed, worker_id=worker, worker_episode_id=local_episode,
                               episode_id=worker+local_episode*count,
                               constructor_seed=base_seed+worker*1009,
                               reset_seed=(base_seed+worker*1009+local_episode*100003) % (2**32),
                               success=run.terminal_success(info, config['tasks'][task]['success_metric']),
                               reward=rewards[worker], length=lengths[worker],
                               legacy_truncated=bool(info['_truncated']),
                               reason=str(info.get('terminated_reason', 'unspecified')),
                               initial=initials[worker], diagnostics=metrics.terminal_diagnostics(info))
                    partial.write(json.dumps(metrics.json_value(row), ensure_ascii=False)+'\n')
                    rows.append(row)
                    completed[worker] += 1
                    previous[worker].fill(0)
                    hidden[:, worker] = 0
                    rewards[worker], lengths[worker] = 0., 0
                    if completed[worker] < quotas[worker]:
                        initials[worker] = initial_state(packet)
                now = time.monotonic()
                if now-last_report >= 20 or len(rows)-last_count >= 10 or len(rows) == args.episodes:
                    progress = dict(completed=len(rows), requested=args.episodes,
                                    successes=sum(row['success'] for row in rows), seconds=now-start,
                                    worker_completed=completed.copy(), worker_quotas=quotas)
                    write_json(output/'progress.json', progress, metrics)
                    print('[progress] {} {}/{} successes={}'.format(
                        task, len(rows), args.episodes, progress['successes']), flush=True)
                    last_report, last_count = now, len(rows)
        rows.sort(key=lambda row: row['episode_id'])
        if [row['episode_id'] for row in rows] != list(range(args.episodes)):
            raise RuntimeError('Missing or duplicate canonical episodes')
        with (output/'episodes.jsonl').open('x') as final:
            for row in rows:
                final.write(json.dumps(metrics.json_value(row), ensure_ascii=False)+'\n')
        result = metrics.summarize(rows, config['tasks'][task]['success_metric'], args.episodes)
        result.update(protocol=task_protocol, policy_role='visual_student',
                      success_definition=config['tasks'][task].get('success_definition'),
                      evaluation_revision=config.get('evaluation_revision'),
                      seed=args.seed, stage=config['stage'], camera=config['vision'],
                      wilson_95=run.wilson(result['successes'], len(rows)),
                      seconds=time.monotonic()-start)
        write_json(output/'summary.json', result, metrics)
        (output/'COMPLETE').write_text('All fixed worker quotas completed\n')
        print('[complete] {} {}/{} = {:.2%}'.format(task, result['successes'], len(rows), result['success_rate']), flush=True)
        return result
    finally:
        if pool is not None:
            pool.close()


def parser_for_cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', choices=['future', 'contact'], required=True)
    p.add_argument('--checkpoint', type=Path, help='Default: Future last 10k / Contact best 6k')
    p.add_argument('--tasks', choices=TASKS, nargs='+', default=list(TASKS))
    p.add_argument('--num-envs', type=int, default=8, help='Parallel environments PER task; tasks run sequentially')
    p.add_argument('--episodes', type=int, default=300, help='Total episodes per task, not per environment')
    p.add_argument('--seed', type=int, default=20261029, help='One master seed; worker i uses seed offset i*1009')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--gl-backend', choices=['egl', 'glfw', 'osmesa'], default='egl')
    p.add_argument('--egl-device-id', type=int, default=0)
    p.add_argument('--inference-batch-size', type=int, default=0,
                   help='0 batches all active environments; 1 uses per-environment forward calls')
    p.add_argument('--worker-timeout', type=int, default=180)
    p.add_argument('--output', type=Path, required=True, help='New output directory; existing protocols are never overwritten')
    return p


def main():
    p = parser_for_cli()
    args = p.parse_args()
    if min(args.num_envs, args.episodes, args.worker_timeout) < 1 or args.inference_batch_size < 0:
        p.error('Positive num-envs/episodes/timeout and nonnegative inference-batch-size required')
    if not 0 <= args.seed < 2**32 or args.egl_device_id < 0 or len(set(args.tasks)) != len(args.tasks):
        p.error('Invalid seed/device or duplicate tasks')
    if (args.output/'protocol.json').exists():
        p.error('Output already contains a protocol; choose a new --output')
    os.environ['MUJOCO_GL'] = args.gl_backend
    os.environ['MUJOCO_EGL_DEVICE_ID'] = str(args.egl_device_id)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    import numpy as np
    import torch
    audit_path, audit, prefix = audit_for(args.variant)
    checkpoint = (args.checkpoint or Path(audit['checkpoint'])).resolve()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved.get('schema') != 'distillation_aux_'+args.variant+'_v1':
        raise ValueError('Checkpoint belongs to a different model variant')
    expected_manifest = {item['path']: item['expected'] for item in audit['files']}
    if saved['metadata']['source_asset_sha256'] != expected_manifest:
        raise ValueError('Checkpoint source manifest differs from the audited historical source')
    source = restore_and_verify(audit, prefix)
    sys.path.insert(0, str(source))
    import mujoco
    run = importlib.import_module('distillation_aux_'+args.variant+'.run')
    metrics = importlib.import_module('distillation.metrics')
    from distillation.core import ActionAdapter
    from distillation.environments import stack_tree
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in this Python process')
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    run.seed_all(args.seed)
    config = copy.deepcopy(saved['config'])
    config['device'], config['seed'] = args.device, args.seed
    config['vision']['gl_backend'] = args.gl_backend
    student = run.make_student(config).eval().requires_grad_(False)
    student.load_state_dict(saved['student'], strict=True)
    if not student.normalizer_fitted.item() or not all(torch.isfinite(v).all() for v in student.state_dict().values()):
        raise RuntimeError('Invalid normalizer or non-finite checkpoint')
    protocol = dict(variant=args.variant, checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                    iteration=saved['iteration'], tasks=args.tasks, master_seed=args.seed,
                    requested_num_envs=args.num_envs, episodes_per_task=args.episodes,
                    max_policy_batch_size=args.inference_batch_size or min(args.episodes, args.num_envs),
                    allocation='fixed near-equal quota per worker; no fastest-episode selection',
                    source_audit=str(audit_path), source_audit_sha256=sha(audit_path),
                    source_root=str(source), source_exceptions=audit['exceptions'],
                    evaluator_sha256=sha(__file__), capture_hook_sha256=sha(Path(__file__).with_name('distill_eval_capture.py')),
                    device=args.device, gl_backend=args.gl_backend, egl_device_id=args.egl_device_id,
                    deterministic_algorithms=True, tf32=False,
                    versions=dict(torch=torch.__version__, numpy=np.__version__, mujoco=mujoco.__version__, python=sys.version),
                    comparison_note='Changing num-envs changes worker seed streams and sampled environment parameters; batching can change floating-point results.')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'protocol.json').open('x') as file:
        file.write(json.dumps(protocol, indent=2))
    write_json(args.output/'evaluation_config.json', config, metrics)
    results = {}
    for task in args.tasks:
        results[task] = evaluate_task(run, config, student, ActionAdapter(config['tasks'][task]), args,
                                     task, args.output/task, protocol, metrics, stack_tree)
        write_json(args.output/'student_evaluation.json', results, metrics)
    if 'model_experiments.run' in sys.modules:
        raise RuntimeError('Unused historical training launcher was imported')
    (args.output/'EVALUATION_COMPLETE').write_text('All requested vectorized task evaluations complete\n')
    rates = [result['success_rate'] for result in results.values()]
    print('[summary] mean={:.2%} minimum={:.2%}; {}'.format(sum(rates)/len(rates), min(rates), args.output), flush=True)


if __name__ == '__main__':
    main()
