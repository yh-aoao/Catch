"""Evaluate the Roll Catch teacher with paired seeds and save physical traces."""
import argparse
import json
from pathlib import Path
import random
import time
from contextlib import ExitStack

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--object', choices=['tennis_ball', 'baseball'], default='tennis_ball')
    parser.add_argument('--speed', type=float, default=None)
    parser.add_argument('--wait-guard', action='store_true')
    parser.add_argument('--collision', choices=['mesh', 'sphere', 'auto'], default='mesh')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--variants', nargs='+', choices=['baseline', 'no_damping', 'long', 'long_no_damping'],
                        default=['baseline', 'no_damping', 'long', 'long_no_damping'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.viewer and len(args.variants) != 1:
        parser.error('Use one variant per viewer process')
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    from distillation.core import observation_array, ActionAdapter
    from distillation.teachers import ROOT, Teacher
    from gym_dcmm.envs.object_test import make_object_test

    torch.set_num_threads(1)
    config = json.loads((ROOT / 'distillation/configs/three_tasks.json').read_text())['tasks']['roll']
    config['checkpoint'] = str(Path(args.checkpoint).resolve())
    teacher, adapter = Teacher(config), ActionAdapter(config)
    summaries = {}
    for variant in args.variants:
        random.seed(args.seed)
        np.random.seed(args.seed)
        env = make_object_test('ycb:' + args.object, task='Catching', object_motion='roll',
                               roll_speed=args.speed, roll_wait_guard=args.wait_guard,
                               ycb_collision=args.collision,
                               object_eval=False, viewer=False, render_mode=None,
                               env_time=5.0 if variant.startswith('long') else 2.5,
                               ycb_joint_damping=0.0 if 'no_damping' in variant else None,
                               steps_per_policy=20)
        episodes = []
        window = None
        stack = ExitStack()
        if args.viewer:
            from ycb_objects.viewer import passive_viewer
            window = stack.enter_context(passive_viewer(env.Dcmm.model, env.Dcmm.data))
            window.cam.lookat[:] = [0, 1.3, .6]
            window.cam.distance = 4
            window.cam.azimuth = 180
            window.cam.elevation = -25
        try:
            for seed in range(args.seed, args.seed + args.episodes):
                random.seed(seed)
                np.random.seed(seed)
                obs, _ = env.reset()
                model, data = env.Dcmm.model, env.Dcmm.data
                bid = int(model.geom_bodyid[env.object_id])
                jid = model.body_jntadr[bid]
                q, v = model.jnt_qposadr[jid], model.jnt_dofadr[jid]
                initial_position, initial_velocity = env.object_pos3d.copy(), env.object_vel6d.copy()
                trace = []
                guard_trace = []
                teacher.parity_check(observation_array(obs)[None])
                start = time.monotonic()
                for _ in range(250):
                    action = teacher.predict(observation_array(obs)[None])[0]
                    obs, _, terminated, truncated, info = env.step(adapter.action_dict(action))
                    trace.append([float(data.time), *data.qpos[q:q+3].tolist(),
                                  *data.qvel[v:v+6].tolist()])
                    if args.wait_guard:
                        guard_trace.append(dict(time=float(data.time),
                                                **info['roll_wait_guard']))
                    if window:
                        window.sync()
                        time.sleep(max(0, start + float(info['env_time']) - time.monotonic()))
                    if terminated or truncated:
                        break
                else:
                    raise RuntimeError('Episode did not terminate within 250 policy steps')
                row = dict(seed=seed, initial_position=initial_position.tolist(),
                           initial_velocity=initial_velocity.tolist(),
                           final_position=data.qpos[q:q+3].tolist(),
                           final_speed=float(np.linalg.norm(data.qvel[v:v+3])),
                           elapsed=float(info['env_time']), reason=env.terminated_reason,
                           terminated=bool(terminated), truncated=bool(truncated),
                           reached_front_edge=any(t[2] < .9 - model.geom_size[env.object_id, 0] for t in trace),
                           success=bool(info.get('roll_eval_success', False)), trace=trace,
                           guard_trace=guard_trace)
                episodes.append(row)
                print(variant, seed, 'y=', round(row['final_position'][1], 3),
                      'speed=', round(row['final_speed'], 3), 'reason=', row['reason'], flush=True)
                (args.output / (variant + '.json')).write_text(json.dumps(episodes, indent=2))
        finally:
            stack.close()
            env.close()
        summaries[variant] = dict(episodes=len(episodes),
                                 object=args.object, speed=args.speed, wait_guard=args.wait_guard,
                                 collision=args.collision,
                                 reached_front_edge=sum(e['reached_front_edge'] for e in episodes),
                                 successes=sum(e['success'] for e in episodes),
                                 ended_on_table=sum(e['final_position'][1] > .9 and e['final_position'][2] > .65
                                                    for e in episodes),
                                 checkpoint_sha256=teacher.hash)
        (args.output / 'summary.json').write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
