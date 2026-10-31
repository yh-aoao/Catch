"""Small, independent initial-depth diagnostic after the formal evaluation.

Example (dcmm Python, GPU/EGL access required only when actually running):
  python analysis/diagnose_distill_depth_pairing_20261029.py

The driver launches the two historical variants in separate processes. Each uses
the original Pool(1), B=1 policy, constructor/reset seeds and current formal
deterministic settings. Only initial observations/derived arrays are copied.
No physics state, RNG, training code or formal evaluation artifact is modified.
"""
import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path('/home/isee/Catch')
BASE = ROOT / 'analysis_artifacts/distill_best_multiseed_20261029_v1'
AUDIT = BASE / 'source_audit.json'
ARRAY_FIELDS = ('cam_fovy', 'cam_xpos', 'cam_xmat', 'geom_xpos', 'geom_xmat',
                'geom_rgba', 'geom_pos', 'geom_quat', 'geom_type')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_env(**kwargs):
    """Read-only metadata extension around the existing independent capture hook."""
    import numpy as np
    from distill_eval_capture import make_env as physical_capture
    env = physical_capture(**kwargs)
    original_reset = env.reset

    def reset(*args, **kw):
        obs, info = original_reset(*args, **kw)
        model, data = env.Dcmm.model, env.Dcmm.data
        render = {
            'extent': float(model.stat.extent),
            'znear': float(model.vis.map.znear),
            'zfar': float(model.vis.map.zfar),
            'simulation_time': float(data.time),
        }
        arrays = {name: np.asarray(getattr(model if name in
                  ('cam_fovy', 'geom_rgba', 'geom_pos', 'geom_quat', 'geom_type')
                  else data, name)).copy() for name in ARRAY_FIELDS}
        hashes = {}
        for name, array in arrays.items():
            hashes[name] = hashlib.sha256(array.tobytes()).hexdigest()
            render[name] = array.tolist()
        render['array_sha256'] = hashes
        obs['_depth_diagnostic_render_json'] = json.dumps(render, sort_keys=True)
        return obs, info

    env.reset = reset
    return env


def read_metadata(packet, key):
    return json.loads(str(packet['teacher_obs'][key][0]))


def save_initial(output, episode, packet, run, config, seed):
    import numpy as np
    physical = read_metadata(packet, '_evaluation_initial_json')
    render = read_metadata(packet, '_depth_diagnostic_render_json')
    depth = np.asarray(packet['depth'][0]).copy()
    proprio = run.proprioception(packet['teacher_obs'], config['stage'])[0].copy()
    arrays = {name: np.asarray(render.pop(name)) for name in ARRAY_FIELDS}
    name = f'episode_{episode:03d}'
    np.savez_compressed(output / (name + '.npz'), depth=depth, proprio=proprio, **arrays)
    row = dict(episode_id=episode, seed=seed,
               reset_seed=(seed + 1000000 + episode * 100003) % (2**32),
               physical=physical, render=render,
               depth_sha256=hashlib.sha256(depth.tobytes()).hexdigest(),
               proprio_sha256=hashlib.sha256(proprio.tobytes()).hexdigest(),
               arrays=name + '.npz')
    (output / (name + '.json')).write_text(json.dumps(row, indent=2))


def worker(args):
    import numpy as np
    audit = json.loads(AUDIT.read_text())[args.variant]
    source = Path(audit['source_root'])
    for item in audit['files']:
        if sha(source / item['path']) != item['actual']:
            raise RuntimeError('Historical source changed: ' + item['path'])
    for item in audit.get('additional_required_assets', []):
        if sha(source / item['path']) != item['sha256']:
            raise RuntimeError('Historical asset changed: ' + item['path'])
    if sha(audit['checkpoint']) != audit['checkpoint_sha256']:
        raise RuntimeError('Checkpoint changed')
    sys.path.insert(0, str(source))
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['MUJOCO_EGL_DEVICE_ID'] = str(args.egl_device_id)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    import torch
    import mujoco
    run = importlib.import_module('distillation_aux_' + args.variant + '.run')
    from distillation.core import ActionAdapter
    from distillation_visual_aux.core import camera_specs
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    saved = torch.load(audit['checkpoint'], map_location='cpu', weights_only=True)
    if saved['schema'] != run.SCHEMA:
        raise RuntimeError('Checkpoint schema mismatch')
    config = copy.deepcopy(saved['config'])
    config['device'] = args.device
    config['vision']['gl_backend'] = 'egl'
    run.seed_all(123)
    student = run.make_student(config).eval().requires_grad_(False)
    student.load_state_dict(saved['student'], strict=True)
    if not student.normalizer_fitted.item():
        raise RuntimeError('Proprioception normalizer is not fitted')
    if not all(torch.isfinite(v).all() for v in student.state_dict().values()):
        raise RuntimeError('Non-finite model weights')
    adapter = ActionAdapter(config['tasks']['throw'])
    output = args.output / args.variant
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(variant=args.variant, checkpoint=audit['checkpoint'],
                    checkpoint_sha256=audit['checkpoint_sha256'], iteration=saved['iteration'],
                    source_audit_sha256=sha(AUDIT), script_sha256=sha(__file__),
                    capture_hook_sha256=sha(Path(__file__).with_name('distill_eval_capture.py')),
                    task='throw', seed=args.seed, episodes=args.episodes,
                    device=args.device, egl_device_id=args.egl_device_id,
                    deterministic_algorithms=True, tf32=False, policy_batch_size=1,
                    camera_specs=camera_specs(config['vision']),
                    depth_encoding='float16((clip(metric_depth_m, near, far)-near)/(far-near))',
                    versions=dict(torch=torch.__version__, numpy=np.__version__,
                                  mujoco=mujoco.__version__, cudnn=torch.backends.cudnn.version(),
                                  python=sys.version))
    protocol_path = output / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise RuntimeError('Existing diagnostic uses a different protocol; choose a new --output')
        if (output / 'COMPLETE').exists():
            print('Already complete:', args.variant, flush=True)
            return
        raise RuntimeError('Incomplete diagnostic found; choose a new --output (no files overwritten)')
    protocol_path.write_text(json.dumps(protocol, indent=2))
    cfg = run.env_config(config, 'throw')
    cfg['extra_labels'] = False
    cfg['env_factory'] = Path(__file__).stem + ':make_env'
    pool = run.Pool('throw', cfg, 1, args.seed + 1000000)
    previous = np.zeros((1, student.action_dim), np.float32)
    hidden = student.initial_state(1)
    episode, length, reward_sum = 0, 0, 0.
    rows = []
    start = time.monotonic()
    try:
        save_initial(output, episode, pool.obs, run, config, args.seed)
        while episode < args.episodes:
            proprio = run.proprioception(pool.obs['teacher_obs'], config['stage'])
            with torch.inference_mode():
                action, hidden, _ = student(
                    torch.as_tensor(pool.obs['depth'], device=args.device)[:, None],
                    torch.as_tensor(proprio, device=args.device)[:, None],
                    torch.as_tensor(previous, device=args.device)[:, None], hidden)
                command = adapter.executable(action[:, 0].cpu().numpy())
            _, reward, done, infos = pool.step(adapter.action_dict(command))
            previous = command.copy()
            length += 1
            reward_sum += float(reward[0])
            if length > config['max_episode_steps']:
                raise RuntimeError('Episode exceeded original max_episode_steps')
            if done[0]:
                info = infos[0]
                rows.append(dict(episode_id=episode, length=length, reward=reward_sum,
                                 success=bool(run.terminal_success(info, config['tasks']['throw']['success_metric'])),
                                 reason=str(info.get('terminated_reason', 'unspecified'))))
                episode += 1
                print('[depth-diagnostic]', args.variant, f'{episode}/{args.episodes}', flush=True)
                previous.fill(0)
                hidden = student.initial_state(1)
                length, reward_sum = 0, 0.
                if episode < args.episodes:
                    save_initial(output, episode, pool.obs, run, config, args.seed)
        if 'model_experiments.run' in sys.modules:
            raise RuntimeError('Unavailable historical launcher unexpectedly imported')
        (output / 'episodes.json').write_text(json.dumps(rows, indent=2))
        (output / 'COMPLETE').write_text(f'{len(rows)} episodes completed in {time.monotonic()-start:.3f}s\n')
    finally:
        pool.close()


def difference(a, b):
    import numpy as np
    if a.shape != b.shape:
        return dict(shape_equal=False, left_shape=list(a.shape), right_shape=list(b.shape))
    values = np.abs(a.astype(np.float64) - b.astype(np.float64))
    changed = values[a != b]
    levels = [0, .5, .9, .95, .99, 1]
    return dict(shape_equal=True, values=int(values.size), different=int(np.count_nonzero(a != b)),
                max_abs=float(values.max()) if values.size else 0.,
                mean_abs=float(values.mean()) if values.size else 0.,
                quantiles_all={str(q): float(v) for q, v in zip(levels, np.quantile(values, levels))},
                quantiles_changed={str(q): float(v) for q, v in zip(levels, np.quantile(changed, levels))}
                if changed.size else {})


def compare(args):
    import numpy as np
    left, right = args.output / 'belief', args.output / 'future'
    protocols = [json.loads((folder / 'protocol.json').read_text()) for folder in (left, right)]
    if any(not (folder / 'COMPLETE').exists() for folder in (left, right)):
        raise RuntimeError('Both diagnostic variants must complete before comparison')
    if protocols[0]['seed'] != protocols[1]['seed'] or protocols[0]['episodes'] != protocols[1]['episodes']:
        raise RuntimeError('Unpaired diagnostic protocols')
    if protocols[0]['camera_specs'] != protocols[1]['camera_specs']:
        raise RuntimeError('Depth camera encodings differ; cannot directly decode metric deltas')
    specs = protocols[0]['camera_specs']
    rows, joined = [], {s['name']: [[], []] for s in specs}
    for episode in range(protocols[0]['episodes']):
        name = f'episode_{episode:03d}'
        meta = [json.loads((folder / (name + '.json')).read_text()) for folder in (left, right)]
        with np.load(left / (name + '.npz'), allow_pickle=False) as a, np.load(right / (name + '.npz'), allow_pickle=False) as b:
            cameras = {}
            for i, spec in enumerate(specs):
                da, db = a['depth'][i], b['depth'][i]
                scale = float(spec['far'] - spec['near'])
                cameras[spec['name']] = dict(encoded=difference(da, db),
                    decoded_clipped_metres=difference(da.astype(np.float64)*scale,
                                                      db.astype(np.float64)*scale))
                joined[spec['name']][0].append(da.reshape(-1).copy())
                joined[spec['name']][1].append(db.reshape(-1).copy())
            rows.append(dict(episode_id=episode,
                physical_hash_equal=meta[0]['physical']['physical_hash'] == meta[1]['physical']['physical_hash'],
                depth_hash_equal=meta[0]['depth_sha256'] == meta[1]['depth_sha256'],
                proprio=difference(a['proprio'], b['proprio']), cameras=cameras,
                render_scalars_equal={key: meta[0]['render'][key] == meta[1]['render'][key]
                                      for key in ('extent', 'znear', 'zfar', 'simulation_time')},
                extra_arrays={key: difference(a[key], b[key]) for key in ARRAY_FIELDS}))
    pooled = {}
    for spec in specs:
        a, b = [np.concatenate(parts) for parts in joined[spec['name']]]
        scale = float(spec['far'] - spec['near'])
        pooled[spec['name']] = dict(encoded=difference(a, b),
            decoded_clipped_metres=difference(a.astype(np.float64)*scale, b.astype(np.float64)*scale))
    result = dict(seed=protocols[0]['seed'], episodes=protocols[0]['episodes'],
        physical_pairs_equal=sum(row['physical_hash_equal'] for row in rows),
        depth_pairs_equal=sum(row['depth_hash_equal'] for row in rows),
        protocols=protocols, per_episode=rows, pooled_cameras=pooled,
        limitations=[
            'Depth arrays are policy packet float16 values, dimensionless and clipped to [0,1].',
            'Decoded metric deltas multiply encoded deltas by each camera far-near. They compare quantized, clipped depths, not original unquantized renderer frames.',
            'Pixels at 0 or 1 are saturated; their original metric difference cannot be recovered. Equal encoded values may conceal sub-quantization differences.',
            'This diagnostic repeats only a few initial frames and is separate from the formal success-rate sample. It cannot determine the effect of image deviations on success by itself.',
            'Physical hashes cover the existing hook fields; extra captured camera/geometry arrays are compared separately. No claim is made that all possible simulator state has been hashed.',
        ])
    (args.output / 'comparison.json').write_text(json.dumps(result, indent=2))
    lines = [f'初始深度配对诊断：seed={result["seed"]}，每模型 {result["episodes"]} 回合。', '',
             f'原物理哈希一致 {result["physical_pairs_equal"]}/{result["episodes"]}；初始编码深度哈希一致 {result["depth_pairs_equal"]}/{result["episodes"]}。', '',
             '深度是策略实际接收的 float16 编码，范围 [0,1]，无量纲。按每个相机的 far-near 换算为裁剪、量化后的米制偏差；不能恢复原始渲染深度，尤其是饱和像素。', '',
             '| 相机 | 不同像素/总像素 | 编码最大绝对差 | 编码平均绝对差 | 解码后最大差(m) | 解码后平均差(m) |',
             '|---|---:|---:|---:|---:|---:|']
    for camera, value in pooled.items():
        e, m = value['encoded'], value['decoded_clipped_metres']
        lines.append(f'| {camera} | {e["different"]}/{e["values"]} | {e["max_abs"]:.9g} | {e["mean_abs"]:.9g} | {m["max_abs"]:.9g} | {m["mean_abs"]:.9g} |')
    lines += ['', 'comparison.json 包含每回合、每相机全部像素与仅变化像素的绝对差分位数，以及本体感知、相机和几何派生数组的比较。原始策略输入数组保存在两模型子目录的 NPZ 文件。', '',
              '这是独立小样本诊断，不纳入正式成功率。哈希不同本身不能证明偏差很小；即使测得偏差小，也不能单凭此实验断定闭环策略不受影响。', '']
    (args.output / 'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({key: result[key] for key in ('seed', 'episodes', 'physical_pairs_equal', 'depth_pairs_equal')}, indent=2))
    print('Saved:', args.output / 'REPORT.md', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=20261029)
    parser.add_argument('--episodes', type=int, default=5, help='Per model, 1-15; at most 30 total episodes')
    parser.add_argument('--output', type=Path, default=BASE / 'depth_diagnostic')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--egl-device-id', type=int, default=0)
    parser.add_argument('--compare-only', action='store_true', help='Read saved NPZ/JSON only; no GPU import or evaluation')
    parser.add_argument('--variant', choices=['belief', 'future'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.episodes <= 15:
        parser.error('--episodes must be 1-15 per model (30 total maximum)')
    args.output = args.output.resolve()
    if args.compare_only:
        compare(args)
    elif args.variant:
        if not (BASE / 'EVALUATION_COMPLETE').exists():
            raise RuntimeError('Wait for all formal evaluations to complete before running this GPU diagnostic')
        worker(args)
    else:
        if not (BASE / 'EVALUATION_COMPLETE').exists():
            raise RuntimeError('Wait for all formal evaluations to complete before running this GPU diagnostic')
        args.output.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, MPLCONFIGDIR='/tmp/catch-mpl', XDG_CACHE_HOME='/tmp/catch-eval-cache',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        for variant in ('belief', 'future'):
            command = [sys.executable, str(Path(__file__).resolve()), '--variant', variant,
                       '--seed', str(args.seed), '--episodes', str(args.episodes), '--output', str(args.output),
                       '--device', args.device, '--egl-device-id', str(args.egl_device_id)]
            with (args.output / (variant + '.log')).open('a', buffering=1) as log:
                subprocess.run(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        compare(args)


if __name__ == '__main__':
    main()
