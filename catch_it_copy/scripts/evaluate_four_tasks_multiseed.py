"""Five evaluation seeds x 100 episodes per target for Roll/Bounce/Throw/Basket."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', nargs='+', choices=['roll', 'bounce', 'throw', 'basket'],
                   default=['roll', 'bounce', 'throw', 'basket'])
    p.add_argument('--seeds', nargs=5, type=int, default=[41000, 42000, 43000, 44000, 45000])
    p.add_argument('--episodes-per-seed', type=int, default=100)
    p.add_argument('--instances', type=int, default=4,
                   help='Creation-time physical draws per target PER SEED for Roll/Bounce/Throw')
    p.add_argument('--jobs', type=int, default=4, help='Concurrent task/seed workers; each runs its targets sequentially')
    p.add_argument('--bounce-targets', choices=['five', 'six'], default='five')
    p.add_argument('--roll-checkpoint', type=Path, default=ROOT.parent/'11-22/roll_catch_best_reward_403.15.pth')
    p.add_argument('--bounce-checkpoint', type=Path, default=ROOT.parent/'11-23/bounce_catch_best_reward_16.15.pth')
    p.add_argument('--throw-checkpoint', type=Path, default=ROOT.parent/'11-19/throw_catch_best_reward_515.51.pth')
    p.add_argument('--basket-checkpoint', type=Path, default=ROOT/'basket_mobile_bridge/runs/tennis_bridge_seed123/teacher_last.pth')
    p.add_argument('--output', type=Path)
    p.add_argument('--resume', action='store_true', help='Skip COMPLETE seed runs with identical request/checkpoints/code')
    p.add_argument('--dry-run', action='store_true', help='Print commands without running evaluations')
    a = p.parse_args()
    if len(set(a.seeds)) != 5 or any(s < 0 or s + a.episodes_per_seed >= 2**32 for s in a.seeds):
        p.error('Supply five distinct valid nonnegative seeds')
    if len(set(a.tasks)) != len(a.tasks) or min(a.episodes_per_seed, a.instances, a.jobs) < 1:
        p.error('Use unique tasks and positive counts')
    if a.episodes_per_seed % a.instances and any(t != 'basket' for t in a.tasks):
        p.error('episodes-per-seed must be divisible by instances')
    checkpoints = {t: getattr(a, t+'_checkpoint').resolve() for t in a.tasks}
    for path in checkpoints.values():
        if not path.is_file():
            p.error('Missing checkpoint: '+str(path))
    output = (a.output or ROOT/'evaluation_results'/datetime.now().strftime('four_tasks_5seed_%Y%m%d_%H%M%S_%f')).resolve()
    manifest = [dict(name=t, motion=t, physics='dr', checkpoint=str(checkpoints[t]),
                     **({'roll_palm_version': 'palm_z_v1'} if t == 'roll' else {}))
                for t in a.tasks if t in ('roll', 'bounce')]
    catalog_files = {t: ROOT/'ycb_objects/calibration_data'/(
        'object_benchmark_creation_v2.json' if t == 'bounce' and a.bounce_targets == 'six'
        else 'object_benchmark_creation_v3.json') for t in a.tasks if t in ('roll', 'bounce')}
    targets = {t: list(json.loads(path.read_text())[t]) for t, path in catalog_files.items()}
    targets.update(throw=['sphere', 'box', 'cylinder', 'ellipsoid', 'capsule'], basket=['tennis_ball'])
    source_paths = {Path(__file__).resolve(), ROOT/'scripts/_eval_basket_d060_seed.py',
                    ROOT/'scripts/evaluate_multitarget_catch.py', ROOT/'scripts/_eval_multitarget_worker.py',
                    ROOT/'scripts/evaluate_throw_generalization.py', ROOT/'train_DCMM.py'}
    # Include transitive project implementations/configuration to guard resume.
    for folder in ('gym_dcmm', 'configs', 'ycb_objects', 'basket_autonomous',
                   'basket_mobile_curriculum', 'basket_mobile_bridge', 'basket_mobile_bridge_large'):
        source_paths.update((ROOT/folder).rglob('*.py'))
    source_paths.update((ROOT/'configs').rglob('*.yaml'))
    spec = dict(tasks=a.tasks, seeds=a.seeds, episodes_per_seed=a.episodes_per_seed,
                instances=a.instances, bounce_targets=a.bounce_targets,
                checkpoints={t: dict(path=str(path), sha256=sha(path)) for t, path in checkpoints.items()},
                catalogs={t: dict(path=str(path), sha256=sha(path)) for t, path in catalog_files.items()},
                sources={str(path.relative_to(ROOT)): sha(path) for path in sorted(source_paths)},
                basket_physical_hoop_diameter_m=.6)
    cases = []
    for task in a.tasks:
        for seed in a.seeds:
            folder = output/task/('seed_'+str(seed))
            common = ['--checkpoint', str(checkpoints[task]), '--episodes', str(a.episodes_per_seed),
                      '--seed', str(seed), '--output', str(folder)]
            if task in ('roll', 'bounce'):
                cmd = [sys.executable, str(ROOT/'scripts/evaluate_multitarget_catch.py'),
                       '--models-file', str(output/'models.json'), '--only', task,
                       '--catalog', str(catalog_files[task]), '--instances', str(a.instances), '--jobs', '1']+common
            elif task == 'throw':
                cmd = [sys.executable, str(ROOT/'scripts/evaluate_throw_generalization.py'),
                       '--instances', str(a.instances)]+common
            else:
                cmd = [sys.executable, str(ROOT/'scripts/_eval_basket_d060_seed.py')]+common
            cases.append((task, seed, folder, cmd))
    total = sum(len(targets[t]) for t in a.tasks)*len(a.seeds)*a.episodes_per_seed
    print('Output:', output, '\nRequested episodes:', total,
          '\nSeeds:', a.seeds, '\nBasket: physical 0.6 m rim, tennis ball, complete far-mobile task', flush=True)
    if a.dry_run:
        for _, _, _, cmd in cases:
            print(shlex.join(cmd))
        return 0
    if a.resume:
        if not (output/'request.json').is_file() or json.loads((output/'request.json').read_text()) != spec:
            p.error('Resume request/model/source mismatch; select a new output directory')
        if json.loads((output/'models.json').read_text()) != manifest:
            p.error('Saved model manifest changed; select a new output directory')
    else:
        if output.exists():
            p.error('Output already exists; use --resume or a new directory')
        output.mkdir(parents=True)
        (output/'request.json').write_text(json.dumps(spec, indent=2))
        (output/'models.json').write_text(json.dumps(manifest, indent=2))
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               MUJOCO_GL=os.environ.get('MUJOCO_GL', 'egl'), WANDB_MODE='disabled',
               PATH=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH', ''))

    def run(case):
        task, seed, folder, cmd = case
        if not (a.resume and (folder/'COMPLETE').is_file()):
            if folder.exists():
                raise RuntimeError('Incomplete seed output exists: '+str(folder)+
                                   '; move this seed directory aside before resuming')
            folder.parent.mkdir(parents=True, exist_ok=True)
            log = folder.parent/(folder.name+'.log')
            print('[start]', task, seed, flush=True)
            with log.open('w') as stream:
                result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode or not (folder/'COMPLETE').is_file():
                raise RuntimeError('Evaluation failed: '+str(log))
        records = json.loads((folder/'summary.json').read_text())
        if isinstance(records, dict):
            records = [records]
        rows = []
        for record in records:
            target = record.get('object', record.get('shape'))
            n, k = int(record['episodes']), int(record['successes'])
            if n != a.episodes_per_seed or not 0 <= k <= n:
                raise RuntimeError('Incomplete target: '+str(folder))
            rows.append(dict(task=task, target=target, seed=seed, successes=k,
                             episodes=n, success_rate=k/n))
        if sorted(r['target'] for r in rows) != sorted(targets[task]):
            raise RuntimeError('Wrong target set: '+str(folder))
        print('[complete]', task, seed, sum(r['successes'] for r in rows), '/', sum(r['episodes'] for r in rows), flush=True)
        return rows

    rows, errors = [], []
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        pending = {pool.submit(run, case): case for case in cases}
        for future in as_completed(pending):
            try:
                rows.extend(future.result())
            except Exception as error:
                errors.append(str(error))
                print(error, file=sys.stderr, flush=True)
    (output/'failures.json').write_text(json.dumps(errors, indent=2))
    rows.sort(key=lambda r: (r['task'], r['target'], r['seed']))
    (output/'per_seed.json').write_text(json.dumps(rows, indent=2))
    with (output/'per_seed.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['task', 'target', 'seed', 'successes', 'episodes', 'success_rate'])
        writer.writeheader()
        writer.writerows(rows)
    totals = []
    for task in a.tasks:
        for target in targets[task]:
            selected = [r for r in rows if r['task'] == task and r['target'] == target]
            n, k = sum(r['episodes'] for r in selected), sum(r['successes'] for r in selected)
            if n:
                totals.append(dict(task=task, target=target, seeds_completed=len(selected),
                                   successes=k, episodes=n, success_rate=k/n))
    (output/'summary.json').write_text(json.dumps(totals, indent=2))
    with (output/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['task', 'target', 'seeds_completed', 'successes', 'episodes', 'success_rate'])
        writer.writeheader()
        writer.writerows(totals)
    lines = ['# Four-task evaluation', '',
             'Roll: end retention; Bounce: continuous controlled hold; Throw: original timeout success; Basket: safe physical score in a 0.6 m hoop.',
             'Success conditions differ across tasks. Check COMPLETE before interpreting this as a full run.', '',
             '|Task|Target|Seeds completed|Successes|Rate|', '|---|---|---:|---:|---:|']
    for row in totals:
        lines.append('|{task}|{target}|{seeds_completed}|{successes}/{episodes}|{success_rate:.1%}|'.format(**row))
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    if errors:
        return 1
    assert sum(r['episodes'] for r in totals) == total
    for path, digest in spec['sources'].items():
        assert sha(ROOT/path) == digest, 'Code changed during evaluation: '+path
    for item in spec['checkpoints'].values():
        assert sha(item['path']) == item['sha256'], 'Checkpoint changed during evaluation'
    (output/'COMPLETE').write_text('All task/seed/target quotas completed\n')
    print('Complete:', output/'REPORT.md', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
