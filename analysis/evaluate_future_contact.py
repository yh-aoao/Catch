"""Evaluate the gated Future+Contact student in the current local environment."""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=1500, help='Episodes per task')
    parser.add_argument('--num-envs', type=int, default=32, help='Parallel processes per active task')
    parser.add_argument('--inference-batch-size', type=int, default=0, help='0 batches all active environments')
    parser.add_argument('--worker-timeout', type=int, default=180)
    parser.add_argument('--seed', type=int, default=20261102)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--egl-device-id', type=int, default=0)
    parser.add_argument('--allow-source-changes', action='store_true')
    args = parser.parse_args()
    if min(args.episodes, args.num_envs, args.worker_timeout) < 1 or min(args.egl_device_id, args.inference_batch_size) < 0:
        parser.error('episodes, num-envs and timeout must be positive; device and batch size nonnegative')
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['MUJOCO_EGL_DEVICE_ID'] = str(args.egl_device_id)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    sys.path.insert(0, str(ROOT/'catch_it_copy'))
    import torch
    from distillation_aux_future_contact import run
    from distillation.teachers import sha256
    from distillation import metrics
    from distillation.environments import stack_tree
    from evaluate_aux_parallel import evaluate_task, write_json
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if saved.get('schema') != run.SCHEMA:
        raise ValueError('Expected a gated Future+Contact checkpoint')
    current = run.provenance()
    original = saved['metadata']['source_asset_sha256']
    differences = {p:dict(training=original.get(p),current=current.get(p))
                   for p in sorted(set(original)|set(current)) if original.get(p)!=current.get(p)}
    if differences and not args.allow_source_changes:
        raise ValueError('Source/assets differ; review with --allow-source-changes to evaluate current environment')
    config = copy.deepcopy(saved['config'])
    config.update(device=args.device, seed=args.seed, eval_episodes=args.episodes)
    config['vision']['gl_backend'] = 'egl'
    config['max_eval_steps'] = max(config['max_eval_steps'], args.episodes*config['max_episode_steps']+1)
    adapters = run.validate(config)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    run.seed_all(args.seed)
    student = run.make_student(config).eval().requires_grad_(False)
    student.load_state_dict(saved['student'], strict=True)
    if not student.normalizer_fitted.item() or not all(torch.isfinite(v).all() for v in student.state_dict().values()):
        raise ValueError('Invalid normalizer or non-finite model weights')
    args.output.mkdir(parents=True, exist_ok=False)
    run.save_json(args.output/'evaluation_config.json', dict(
        config=config, iteration=saved['iteration'], checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=sha256(args.checkpoint), source_changes=differences,
        current_source_asset_sha256=current, source_changes_accepted=args.allow_source_changes,
        note='Current local environment; source differences prevent claiming exact training-environment reproduction'))
    protocol = dict(iteration=saved['iteration'], checkpoint_sha256=sha256(args.checkpoint),
                    master_seed=args.seed, requested_num_envs=args.num_envs,
                    episodes_per_task=args.episodes, inference_batch_size=args.inference_batch_size,
                    allocation='fixed near-equal quota per worker',
                    source_changes_accepted=args.allow_source_changes,
                    evaluator_sha256=sha256(__file__),
                    parallel_evaluator_sha256=sha256(Path(__file__).with_name('evaluate_aux_parallel.py')),
                    capture_sha256=sha256(Path(__file__).with_name('distill_eval_capture.py')),
                    comparison_note='Changing num-envs changes sampled seed streams; batching can change floating-point results')
    write_json(args.output/'protocol.json', protocol, metrics)
    print('iteration={} episodes/task={} environments/task={} source differences={}'.format(
        saved['iteration'],args.episodes,args.num_envs,len(differences)),flush=True)
    results = {}
    for task, adapter in adapters.items():
        results[task] = evaluate_task(run, config, student, adapter, args, task,
                                     args.output/task, protocol, metrics, stack_tree)
        write_json(args.output/'student_evaluation.json', results, metrics)
    if any(r['episodes'] != args.episodes for r in results.values()):
        raise RuntimeError('Incomplete evaluation')
    (args.output/'EVALUATION_COMPLETE').write_text('All task quotas completed\n')
    print(json.dumps({t:{k:r[k] for k in ('successes','episodes','success_rate')} for t,r in results.items()},indent=2))


if __name__ == '__main__':
    main()
