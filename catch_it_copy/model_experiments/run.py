"""Run independently versioned experiments sequentially, stopping at first failure."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {
    'baseline': 'distillation_event',
    'event': 'distillation_event',
    'feature': 'distillation_feature',
    'camera_gate': 'distillation_camera_gate',
    'action_heads': 'distillation_action_heads',
    'moe': 'distillation_moe',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['check', 'smoke', 'train'])
    parser.add_argument('--stage', choices=['track', 'catch'], required=True)
    parser.add_argument('--variants', nargs='+', choices=list(PACKAGES), default=list(PACKAGES))
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-envs', type=int, default=8, help='Workers per motion task')
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--eval-every', type=int, default=1000, help='Evaluate every N iterations')
    parser.add_argument('--episodes', type=int, default=200, help='Evaluation episodes per task')
    parser.add_argument('--tag', default='', help='Suffix for the output directory (keeps existing runs)')
    parser.add_argument('--skip-teacher-baseline', action='store_true',
                        help='Skip the initial teacher success-rate evaluation')
    parser.add_argument('--gl-backend', choices=['auto', 'egl', 'glfw', 'osmesa'], default='auto')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.num_envs < 1 or args.iterations < 1 or args.eval_every < 1 or args.episodes < 1:
        parser.error('num-envs, iterations, eval-every and episodes must be positive')
    for variant in args.variants:
        package = PACKAGES[variant]
        config = ROOT / package / 'configs' / (args.stage + ('_baseline' if variant == 'baseline' else '') + '.json')
        name = variant + '_' + args.stage + '_' + args.mode + '_seed' + str(args.seed)
        if args.tag:
            name += '_' + args.tag
        output = ROOT / package / 'runs' / name
        command = [sys.executable, '-m', package + '.run', 'check' if args.mode == 'check' else 'train',
                   '--config', str(config), '--output', str(output), '--seed', str(args.seed),
                   '--device', 'cpu' if args.mode == 'check' else args.device,
                   '--num-envs', str(args.num_envs), '--gl-backend', args.gl_backend]
        if args.mode == 'smoke':
            command += ['--iterations', '2', '--baseline-episodes', '1', '--episodes', '2',
                        '--eval-every', '1', '--allow-zero-teacher-success']
        elif args.mode == 'train':
            command += ['--iterations', str(args.iterations), '--eval-every', str(args.eval_every),
                        '--episodes', str(args.episodes)]
            if args.skip_teacher_baseline:
                command += ['--skip-teacher-baseline']
        print(json.dumps({'variant': variant, 'command': command}, ensure_ascii=False), flush=True)
        if not args.dry_run:
            # check=True prevents starting subsequent expensive jobs after an error.
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
