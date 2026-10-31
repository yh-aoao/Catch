"""Evaluate one Contact checkpoint across three tasks; at most two GPU jobs."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time


ROOT = Path('/home/isee/Catch')
OUT = ROOT/'analysis_artifacts/contact_multiseed_20261030_v1'
PYTHON = '/home/isee/miniconda3/envs/dcmm/bin/python'
SEEDS = [20261029, 20262029, 20263029, 20264029, 20265029]


def main():
    audit = json.loads((OUT/'source_audit.json').read_text())['contact']
    queue = ['throw', 'roll', 'bounce']
    active, completed = {}, []
    (OUT/'formal').mkdir(exist_ok=True)
    plan = dict(variant='contact', checkpoint=audit['checkpoint'],
                checkpoint_sha256=audit['checkpoint_sha256'], iteration=audit['iteration'],
                seeds=SEEDS, episodes_per_seed_per_task=300, expected_total_episodes=4500,
                tasks=queue.copy(), parallel_jobs=2, streams_per_job=5, policy_batch_size=1,
                original_weights_unchanged=True, no_training=True,
                references='distill_best_multiseed_20261029_v1: same seed and episode protocol; different training iteration')
    (OUT/'PLAN.json').write_text(json.dumps(plan, indent=2))
    start = time.monotonic()
    try:
        while queue or active:
            while queue and len(active) < 2:
                task = queue.pop(0)
                name = 'contact_'+task
                folder = OUT/'formal'/name
                if (folder/'COMPLETE').exists():
                    completed.append(name)
                    continue
                log = (OUT/(name+'.log')).open('a', buffering=1)
                command = [PYTHON, str(ROOT/'analysis/evaluate_contact_multiseed_20261030.py'),
                           '--variant','contact','--task',task,'--seeds',*map(str,SEEDS),
                           '--episodes','300','--output',str(folder)]
                env = dict(os.environ, MPLCONFIGDIR='/tmp/catch-mpl', XDG_CACHE_HOME='/tmp/catch-eval-cache',
                           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
                proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, start_new_session=True)
                active[name] = (proc, log)
                print('STARTED', name, 'pid', proc.pid, flush=True)
                time.sleep(3)
            for name, (proc, log) in list(active.items()):
                code = proc.poll()
                if code is not None:
                    log.close()
                    del active[name]
                    if code:
                        (OUT/'FAILED.json').write_text(json.dumps(dict(case=name,returncode=code), indent=2))
                        raise RuntimeError(name+' exited '+str(code)+'; see its .log')
                    completed.append(name)
                    print('FINISHED', name, flush=True)
            (OUT/'RUNNING.json').write_text(json.dumps(dict(seconds=time.monotonic()-start,
                active={name:proc.pid for name,(proc,_) in active.items()}, completed=completed,
                remaining=['contact_'+task for task in queue]), indent=2))
            if queue or active:
                time.sleep(5)
        (OUT/'EVALUATION_COMPLETE').write_text('All three Contact task evaluations completed\n')
        print('ALL CONTACT EVALUATIONS COMPLETE', round(time.monotonic()-start,1), 'seconds', flush=True)
    finally:
        for proc, log in active.values():
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
            log.close()


if __name__ == '__main__':
    main()
