"""Export synchronized top/wrist metric depth under the original teachers.

python -m distillation_visual.camera_pair --stage catch --output ...
This diagnostic does not change the single-camera student architecture.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .run import ROOT, TASK_IDS, legacy, env_config, validate, seed_all, save_json, write_gray_png
from .environments import Pool


def preview(depth, near, far):
    return np.rint(255 * (1 - np.clip((depth-near)/(far-near), 0, 1))).astype(np.uint8)


def export(config, teachers, adapters, output, steps, wrist_near=.01, wrist_far=1.5):
    original = legacy(config['stage'])
    report = {}
    for task, adapter in adapters.items():
        folder = output / task
        folder.mkdir(parents=True, exist_ok=False)
        frames_dir = folder / 'frames'
        frames_dir.mkdir()
        cfg = env_config(config, task)
        cfg['inspection_cameras'] = ['top', 'wrist']
        cfg['vision'] = dict(cfg['vision'], camera='top')
        pool = Pool(task, cfg, 1, config['seed'] + TASK_IDS[task]*10000)
        top, wrist, times, episodes, ends = [], [], [], [], []
        try:
            for i in range(steps):
                packet = pool.obs
                top.append(packet['camera_depth_metres']['top'][0].copy())
                wrist.append(packet['camera_depth_metres']['wrist'][0].copy())
                times.append(float(packet['sim_time'][0]))
                episodes.append(int(packet['episode_id'][0]))
                command = adapter.canonical(teachers[task].predict(original.observation_array(packet['teacher_obs'])))
                _, _, done, _ = pool.step(adapter.action_dict(command))
                ends.append(bool(done[0]))
                if (i+1) % 20 == 0 or i+1 == steps:
                    print('[camera-pair] {} {}/{}'.format(task, i+1, steps), flush=True)
        finally:
            pool.close()
        top, wrist = np.stack(top), np.stack(wrist)
        v = config['vision']
        np.savez_compressed(folder / 'synchronized_depth.npz', top_depth_m=top, wrist_depth_m=wrist,
                            sim_time=np.asarray(times), episode_id=np.asarray(episodes),
                            done_after_action=np.asarray(ends), frame_index=np.arange(steps),
                            top_preview_range_m=np.array([v['near'], v['far']]),
                            wrist_preview_range_m=np.array([wrist_near, wrist_far]))
        selected = set(np.linspace(0, steps-1, min(12, steps), dtype=int).tolist())
        tiles = []
        for i in range(steps):
            pair = np.concatenate([preview(top[i], v['near'], v['far']),
                                   np.zeros((v['height'], 4), np.uint8),
                                   preview(wrist[i], wrist_near, wrist_far)], axis=1)
            # Nearest-neighbor enlargement for inspection only; raw NPZ stays HxW.
            write_gray_png(frames_dir / ('frame_%04d_ep%03d.png' % (i, episodes[i])),
                           np.repeat(np.repeat(pair, 2, axis=0), 2, axis=1))
            if i in selected:
                tiles.append(pair)
        while len(tiles) % 3:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.concatenate([np.concatenate(tiles[i:i+3], axis=1) for i in range(0, len(tiles), 3)], axis=0)
        write_gray_png(folder / 'top_left_wrist_right.png', sheet)
        report[task] = {'frames': steps, 'shape_each_camera': list(top.shape),
                        'layout': 'Each pair: top LEFT, wrist RIGHT; nearer surfaces are brighter.',
                        'contact_sheet_frame_indices': sorted(selected),
                        'episodes_seen': len(set(episodes)), 'completed_episodes': sum(ends),
                        'top_min_m': float(top.min()), 'wrist_min_m': float(wrist.min()),
                        'top_preview_range_m': [v['near'], v['far']],
                        'wrist_preview_range_m': [wrist_near, wrist_far],
                        'raw_depth': 'float32 metres, no application clipping; MuJoCo camera clipping still applies.',
                        'sync': 'Both views rendered before the same action, without advancing simulation.'}
        save_json(folder / 'metadata.json', report[task])
        save_json(output / 'camera_pair_report.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['catch', 'track'], default='catch')
    p.add_argument('--tasks', nargs='+', choices=list(TASK_IDS), default=list(TASK_IDS))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--gl-backend', choices=['auto', 'egl', 'glfw', 'osmesa'], default='auto')
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--wrist-near', type=float, default=.01)
    p.add_argument('--wrist-far', type=float, default=1.5)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if args.steps < 1 or not 0 < args.wrist_near < args.wrist_far:
        p.error('Require steps > 0 and 0 < wrist-near < wrist-far')
    config = json.loads((ROOT / 'distillation_visual/configs' / (args.stage+'.json')).read_text(encoding='utf-8-sig'))
    config.update(device=args.device, seed=args.seed)
    config['vision']['gl_backend'] = args.gl_backend
    config['tasks'] = {t: config['tasks'][t] for t in args.tasks}
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        p.error('CUDA unavailable; select --device cpu')
    adapters = validate(config)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose a new empty output directory for synchronized capture')
    output.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    torch.set_num_threads(1)
    original = legacy(args.stage)
    teachers = original.make_teachers(config)
    checks = original.check(config, teachers)
    save_json(output / 'teacher_check.json', checks)
    if not all(row['interface_ready'] for row in checks.values()):
        raise ValueError('Teacher interface check failed')
    save_json(output / 'capture_config.json', config)
    export(config, teachers, adapters, output, args.steps, args.wrist_near, args.wrist_far)


if __name__ == '__main__':
    main()
