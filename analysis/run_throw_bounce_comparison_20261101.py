"""Complete Throw tests and cross-check Bounce checkpoints under both scenes."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import time

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT/'catch_it_copy'
OUT = ROOT/'analysis_artifacts/throw_bounce_model_comparison_20261031_v1'
PYTHON = '/home/isee/miniconda3/envs/dcmm/bin/python'
RUNNER = ROOT/'analysis/evaluate_three_task_balls_20261031.py'
SAVED = PROJECT/'outputs/bounce_three_ball_catching_v1_seed123/2026-10-28/11:36:32/resolved_config.yaml'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(SAVED)
    cfg.bounce_physics, cfg.bounce_launch = 'legacy', 'legacy'
    cfg.bounce_ball_mode, cfg.bounce_ball_weights, cfg.bounce_material_file = 'fixed', None, None
    legacy = OUT/'bounce_legacy_config.yaml'
    OmegaConf.save(cfg, legacy)
    old = ROOT/'best_model/bounce_catch_best_reward_653.87.pth'
    new = PROJECT/'outputs/bounce_three_ball_catching_v1_seed123/2026-10-28/11:36:32/nn/bounce_catch_best_reward_464.31.pth'
    throw = ROOT/'best_model/throw_catch_best_reward_497.09.pth'
    jobs = [(f'bounce653_physical_{ball}', 'bounce', ball, SAVED, old)
            for ball in ('original', 'tennis_ball', 'baseball')]
    jobs += [('bounce653_legacy_original','bounce','original',legacy,old),
             ('bounce464_legacy_original','bounce','original',legacy,new)]
    jobs += [(f'throw497_{ball}', 'throw', ball, OUT/'throw_config.yaml', throw)
             for ball in ('original', 'tennis_ball', 'baseball')]
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONPATH=str(PROJECT))
    plan = []
    for name, task, ball, config, checkpoint in jobs:
        command = [PYTHON, '-u', str(RUNNER), '--task', task, '--object', ball,
                   '--config', str(config), '--checkpoint', str(checkpoint),
                   '--episodes', '100', '--seed', '0', '--output', str(OUT/name)]
        plan.append(dict(name=name, command=command))
    (OUT/'plan.json').write_text(json.dumps(plan, indent=2))

    def run(job):
        name = job['name']
        if (OUT/name/'COMPLETE').is_file():
            return dict(name=name, returncode=0, reused_complete=True, command=job['command'])
        if (OUT/name).exists():
            raise RuntimeError('Partial result exists; inspect before rerunning: '+name)
        print('START '+name, flush=True)
        started = time.monotonic()
        with (OUT/(name+'.log')).open('x') as log:
            result = subprocess.run(job['command'], cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT)
        record = dict(name=name, returncode=result.returncode, seconds=time.monotonic()-started, command=job['command'])
        print(json.dumps(record), flush=True)
        return record

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(run, plan))
    (OUT/'run_status.json').write_text(json.dumps(results, indent=2))
    if any(r['returncode'] for r in results):
        raise SystemExit('Evaluation failure; inspect per-case logs')
    print('ALL 800 NEW EPISODES COMPLETE', flush=True)


if __name__ == '__main__':
    main()
