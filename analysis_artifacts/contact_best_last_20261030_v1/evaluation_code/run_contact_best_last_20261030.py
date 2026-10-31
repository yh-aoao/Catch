"""Compare Contact best/last using one shared seed and 300 episodes per task."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path('/home/isee/Catch')
OUT = ROOT/'analysis_artifacts/contact_best_last_20261030_v1'
PYTHON = '/home/isee/miniconda3/envs/dcmm/bin/python'
SEED = 20261029


def main():
    audit = json.loads((OUT/'source_audit.json').read_text())
    queue = [(label, task) for task in ['throw','roll','bounce'] for label in ['best','last']]
    active, completed = {}, []
    (OUT/'formal').mkdir(exist_ok=True)
    (OUT/'PLAN.json').write_text(json.dumps(dict(
        variant='contact', models={k:{f:a[f] for f in ['checkpoint','checkpoint_sha256','iteration']} for k,a in audit.items()},
        seeds=[SEED], episodes_per_task_per_checkpoint=300, expected_total_episodes=1800,
        tasks=['throw','roll','bounce'], parallel_jobs=2, streams_per_job=1,
        policy_batch_size=1, user_requested_one_seed=True, no_training=True), indent=2))
    start = time.monotonic()
    try:
        while queue or active:
            while queue and len(active) < 2:
                label,task = queue.pop(0)
                name = label+'_'+task
                folder = OUT/'formal'/name
                if (folder/'COMPLETE').exists():
                    completed.append(name)
                    continue
                log = (OUT/(name+'.log')).open('a', buffering=1)
                command = [PYTHON,str(ROOT/'analysis/evaluate_contact_best_last_20261030.py'),
                           '--variant','contact','--checkpoint-label',label,'--task',task,
                           '--seeds',str(SEED),'--episodes','300','--output',str(folder)]
                env = dict(os.environ, MPLCONFIGDIR='/tmp/catch-mpl', XDG_CACHE_HOME='/tmp/catch-eval-cache',
                           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
                proc = subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[name] = (proc,log)
                print('STARTED',name,'pid',proc.pid,flush=True)
                time.sleep(2)
            for name,(proc,log) in list(active.items()):
                code = proc.poll()
                if code is not None:
                    log.close()
                    del active[name]
                    if code:
                        (OUT/'FAILED.json').write_text(json.dumps(dict(case=name,returncode=code),indent=2))
                        raise RuntimeError(name+' failed; see log')
                    completed.append(name)
                    print('FINISHED',name,flush=True)
            (OUT/'RUNNING.json').write_text(json.dumps(dict(seconds=time.monotonic()-start,
                active={name:proc.pid for name,(proc,_) in active.items()},completed=completed,
                remaining=[label+'_'+task for label,task in queue]),indent=2))
            if queue or active:
                time.sleep(5)
        (OUT/'EVALUATION_COMPLETE').write_text('All 1800 Contact best/last episodes complete\n')
        print('ALL BEST/LAST EVALUATIONS COMPLETE',round(time.monotonic()-start,1),'seconds',flush=True)
    finally:
        for proc,log in active.values():
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
            log.close()


if __name__ == '__main__':
    main()
