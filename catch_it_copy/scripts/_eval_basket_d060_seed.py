"""Evaluate an existing bridge teacher with a physical 0.6 m hoop, no training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    if args.episodes < 1 or args.seed < 0 or args.seed + args.episodes >= 2**32:
        parser.error('Use positive episodes and valid nonnegative seeds')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import numpy as np
    import torch
    from basket_mobile_bridge.policy import Policy, SCHEMA
    from basket_mobile_bridge.run import evaluation_options
    from basket_mobile_bridge_large.environment import AutonomousTeacherEnv
    torch.set_num_threads(1)
    checkpoint = args.checkpoint.resolve()
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved.get('schema') not in (SCHEMA, 'basket_mobile_bridge_large_v1'):
        raise ValueError('Expected a mobile bridge teacher checkpoint')
    options = evaluation_options(saved['config'])
    options.update(hoop_diameter=.6, evaluation_task='far_mobile', fixed_replay_probability=0.)
    model = Policy().cpu()
    model.load_state_dict(saved['model'], strict=True)
    model.eval()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'request.json').write_text(json.dumps(dict(
        checkpoint=str(checkpoint), checkpoint_sha256=digest,
        seed=args.seed, episodes=args.episodes, options=options,
        criterion='task_success: complete far-mobile bridge task with safe physical basket score; no carry-only/fixed rehearsal',
        physical_hoop_diameter_m=.6, object='tennis_ball'), indent=2))
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = AutonomousTeacherEnv(options)
    rows = []
    try:
        assert np.isclose(env.basket_cfg.basket_radius, .3)
        assert np.isclose(env.radius, .033275785333333335)
        # Verify actual rim centers, rather than merely the scoring threshold.
        rim = [i for i in range(env.Dcmm.model.ngeom)
               if (env.Dcmm.model.geom(i).name or '').startswith('basket_seg_')]
        assert len(rim) == 24
        assert np.allclose(np.linalg.norm(env.Dcmm.model.geom_pos[rim, :2], axis=1), .3)
        with torch.inference_mode(), (args.output / 'episodes.jsonl').open('w') as stream:
            for episode in range(args.episodes):
                seed = args.seed + episode
                np.random.seed(seed)
                random.seed(seed)
                torch.manual_seed(seed)
                obs, _ = env.reset(seed=seed)
                total = 0.
                for step in range(512):
                    action = model.act(torch.as_tensor(obs, dtype=torch.float32)[None])[0].numpy()
                    obs, reward, terminated, truncated, info = env.step(action)
                    total += float(reward)
                    if not np.isfinite(total) or not np.isfinite(env.Dcmm.data.qpos).all():
                        raise RuntimeError('Non-finite rollout')
                    if terminated or truncated:
                        break
                else:
                    raise RuntimeError('Episode did not terminate within 512 policy steps')
                assert info['curriculum_task'] == 'mobile'
                success = bool(info['task_success'])
                assert success == bool(info['safe_basket_success'])
                row = dict(object='tennis_ball', episode=episode, episode_seed=seed,
                           success=success, reason=info['terminated_reason'],
                           steps=step+1, reward=total, terminated=bool(terminated), truncated=bool(truncated),
                           reached_throw_phase=bool(info['reached_throw_phase']))
                rows.append(row)
                stream.write(json.dumps(row)+'\n')
                stream.flush()
                print('[basket] {}/{} successes={}'.format(episode+1, args.episodes,
                      sum(r['success'] for r in rows)), flush=True)
    finally:
        env.close()
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
    count = sum(r['success'] for r in rows)
    summary = dict(object='tennis_ball', episodes=len(rows), successes=count,
                   success_rate=count/len(rows), seed=args.seed,
                   checkpoint_sha256=digest, physical_hoop_diameter_m=.6)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    (args.output / 'COMPLETE').write_text('All requested Basket episodes completed\n')


if __name__ == '__main__':
    main()
