"""Audited independent multi-seed evaluation; preserves original one-worker streams.

Five Pool(count=1) streams run concurrently, one per evaluation seed. Each stream
retains the original constructor seed and per-episode reset seed sequence. No
new worker splitting or changes to trained policies/physics/success criteria.
"""
import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['contact'], required=True)
    parser.add_argument('--checkpoint-label', choices=['best', 'last'], required=True)
    parser.add_argument('--task', choices=['throw', 'roll', 'bounce'], required=True)
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--episodes', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.episodes < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Use positive episodes and unique seeds')
    root = Path('/home/isee/Catch')
    audit_path = root/'analysis_artifacts/contact_best_last_20261030_v1/source_audit.json'
    audit = json.loads(audit_path.read_text())[args.checkpoint_label]
    source = Path(audit['source_root'])
    for item in audit['files']:
        if sha(source/item['path']) != item['actual']:
            raise RuntimeError('Evaluation source changed: '+item['path'])
    for item in audit.get('additional_required_assets', []):
        if sha(source/item['path']) != item['sha256']:
            raise RuntimeError('Required historical mesh changed: '+item['path'])
    if sha(audit['checkpoint']) != audit['checkpoint_sha256']:
        raise RuntimeError('Checkpoint changed')
    sys.path.insert(0, str(source))
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['MUJOCO_EGL_DEVICE_ID'] = '0'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    import numpy as np
    import torch
    import mujoco
    run = importlib.import_module('distillation_aux_'+args.variant+'.run')
    from distillation.core import ActionAdapter
    from distillation.environments import stack_tree
    from distillation.metrics import summarize, terminal_diagnostics, json_value
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    saved = torch.load(audit['checkpoint'], map_location='cpu', weights_only=True)
    assert saved['schema'] == run.SCHEMA
    config = copy.deepcopy(saved['config'])
    config['device'] = args.device
    config['vision']['gl_backend'] = 'egl'
    run.seed_all(123)
    student = run.make_student(config).eval().requires_grad_(False)
    student.load_state_dict(saved['student'], strict=True)
    assert student.normalizer_fitted.item()
    assert all(torch.isfinite(v).all() for v in student.state_dict().values())
    adapter = ActionAdapter(config['tasks'][args.task])
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = dict(variant=args.variant, checkpoint_label=args.checkpoint_label, task=args.task, seeds=args.seeds, episodes_per_seed=args.episodes,
                    checkpoint=audit['checkpoint'], checkpoint_sha256=audit['checkpoint_sha256'],
                    iteration=saved['iteration'], source_audit=str(audit_path), source_root=str(source),
                    source_audit_sha256=sha(audit_path), evaluator_sha256=sha(__file__),
                    capture_hook_sha256=sha(Path(__file__).with_name('distill_eval_capture.py')),
                    source_exceptions=audit['exceptions'], device=args.device,
                    batch_streams=len(args.seeds), policy_batch_size=1, deterministic_algorithms=True, tf32=False,
                    constructor_seed='evaluation_seed + 1000000 + TASK_ID*10000',
                    reset_seed='constructor_seed + episode_index*100003',
                    versions=dict(torch=torch.__version__, numpy=np.__version__, mujoco=mujoco.__version__,
                                  cudnn=torch.backends.cudnn.version(), python=sys.version),
                    no_teacher_required=True)
    protocol_path = args.output/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise RuntimeError('Existing output uses a different evaluation protocol')
    protocol_path.write_text(json.dumps(protocol, indent=2))
    pools, records, reward_sum, lengths, initials = {}, {}, {}, {}, {}
    files = {}
    saved_count = {}
    summary = {}
    start = time.monotonic()
    last_report = start
    try:
        for seed in args.seeds:
            path = args.output/f'seed_{seed}.jsonl'
            existing = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
            assert [r['episode_id'] for r in existing] == list(range(len(existing)))
            assert len(existing) <= args.episodes
            records[seed] = existing
            saved_count[seed] = len(existing)
            if len(existing) == args.episodes:
                continue
            cfg = run.env_config(config, args.task)
            cfg['extra_labels'] = False
            cfg['env_factory'] = 'distill_eval_capture:make_env'
            base_seed = seed+1000000+run.TASK_IDS[args.task]*10000
            pools[seed] = run.Pool(args.task, cfg, 1, base_seed)
            reward_sum[seed], lengths[seed] = 0., 0
            initials[seed] = extract_initial(pools[seed].obs)
            files[seed] = path.open('a', buffering=1)
        order = list(pools)
        previous = {seed: np.zeros((1, student.action_dim), np.float32) for seed in order}
        hidden = {seed: student.initial_state(1) for seed in order}
        episode_index = {seed: 0 for seed in order}
        while pools:
            active = list(pools)
            depth = np.concatenate([pools[s].obs['depth'] for s in active])
            proprio = np.concatenate([run.proprioception(pools[s].obs['teacher_obs'], config['stage']) for s in active])
            prior = np.concatenate([previous[s] for s in active])
            state = torch.cat([hidden[s] for s in active], dim=1)
            with torch.inference_mode():
                inputs = (torch.as_tensor(depth, device=args.device)[:,None],
                          torch.as_tensor(proprio, device=args.device)[:,None],
                          torch.as_tensor(prior, device=args.device)[:,None])
                actions, states = [], []
                for i in range(len(active)):
                    one, one_state, _ = student(*(v[i:i+1] for v in inputs), state[:,i:i+1].contiguous())
                    actions.append(one)
                    states.append(one_state)
                action, new_hidden = torch.cat(actions, 0), torch.cat(states, 1)
                command = adapter.executable(action[:,0].cpu().numpy())
            physical = adapter.action_dict(command)
            for i, seed in enumerate(active):
                hidden[seed] = new_hidden[:,i:i+1].contiguous()
                previous[seed] = command[i:i+1].copy()
                pools[seed].pipes[0].send(('step',{k:v[i] for k,v in physical.items()}))
            finished = []
            for seed in active:
                pool = pools[seed]
                packet, reward, done, info = pool._receive(pool.pipes[0])
                pool.obs = stack_tree([packet])
                reward_sum[seed] += float(reward)
                lengths[seed] += 1
                if lengths[seed] > config['max_episode_steps']:
                    raise RuntimeError('Episode exceeded original max_episode_steps')
                if done:
                    metric = config['tasks'][args.task]['success_metric']
                    row = dict(seed=seed, episode_id=episode_index[seed],
                        reset_seed=(seed+1000000+run.TASK_IDS[args.task]*10000+episode_index[seed]*100003) % (2**32),
                        success=run.terminal_success(info, metric), reward=reward_sum[seed], length=lengths[seed],
                        legacy_truncated=bool(info['_truncated']), reason=str(info.get('terminated_reason','unspecified')),
                        initial=initials[seed], diagnostics=terminal_diagnostics(info))
                    if episode_index[seed] < saved_count[seed]:
                        old = records[seed][episode_index[seed]]
                        if old['initial']['physical_hash'] != row['initial']['physical_hash']:
                            raise RuntimeError('Resume initial state mismatch')
                    else:
                        files[seed].write(json.dumps(json_value(row))+'\n')
                        records[seed].append(row)
                    episode_index[seed] += 1
                    previous[seed][:] = 0
                    hidden[seed] = student.initial_state(1)
                    reward_sum[seed], lengths[seed] = 0., 0
                    if len(records[seed]) == args.episodes:
                        finished.append(seed)
                    else:
                        initials[seed] = extract_initial(pool.obs)
            for seed in finished:
                pools.pop(seed).close()
                files.pop(seed).close()
            if time.monotonic()-last_report >= 20 or not pools:
                total = sum(len(v) for v in records.values())
                report = dict(completed=total, requested=args.episodes*len(args.seeds),
                              seconds=time.monotonic()-start,
                              streams={str(s):len(v) for s,v in records.items()},
                              policy_batch_size=1)
                (args.output/'progress.json').write_text(json.dumps(report,indent=2))
                print('[progress]',args.variant,args.task,json.dumps(report),flush=True)
                last_report = time.monotonic()
        for seed in args.seeds:
            summary[str(seed)] = summarize(records[seed], config['tasks'][args.task]['success_metric'], args.episodes)
            summary[str(seed)].pop('records')
        if 'model_experiments.run' in sys.modules:
            raise RuntimeError('Unavailable launcher imported')
        result = dict(protocol=protocol, per_seed=summary, successes=sum(r['success'] for v in records.values() for r in v),
                      episodes=sum(len(v) for v in records.values()), seconds=time.monotonic()-start,
                      policy_batch_size=1)
        (args.output/'summary.json').write_text(json.dumps(result,indent=2))
        (args.output/'COMPLETE').write_text('All canonical seed/episode pairs completed\n')
        print('[complete]',args.variant,args.task,result['successes'],result['episodes'],round(result['seconds'],2),flush=True)
    finally:
        for pool in pools.values():
            pool.close()
        for file in files.values():
            file.close()


def extract_initial(packet):
    initial = json.loads(str(packet['teacher_obs']['_evaluation_initial_json'][0]))
    initial['depth_sha256'] = hashlib.sha256(packet['depth'][0].tobytes()).hexdigest()
    return initial


if __name__ == '__main__':
    main()
