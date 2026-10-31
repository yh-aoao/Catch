"""Run six audited model/task evaluations, at most two GPU jobs concurrently."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/home/isee/Catch')
OUT = ROOT/'analysis_artifacts/distill_best_multiseed_20261029_v1'
PYTHON = '/home/isee/miniconda3/envs/dcmm/bin/python'
SEEDS = [20261029, 20262029, 20263029, 20264029, 20265029]


def main():
    queue = [(model,task) for task in ['throw','roll','bounce'] for model in ['belief','future']]
    active = {}
    completed = []
    (OUT/'formal').mkdir(exist_ok=True)
    (OUT/'PLAN.json').write_text(json.dumps(dict(models=['belief_best_10000','future_last_10000'],
        seeds=SEEDS, episodes_per_seed_per_task=300, expected_total_episodes=9000,
        tasks=['throw','roll','bounce'], parallel_jobs=2, streams_per_job=5,
        policy_batch_size=1,
        original_weights_unchanged=True, no_training=True), indent=2))
    start = time.monotonic()
    while queue or active:
        while queue and len(active)<2:
            model,task = queue.pop(0)
            name = model+'_'+task
            folder = OUT/'formal'/name
            if (folder/'COMPLETE').exists():
                completed.append(name)
                continue
            log = (OUT/(name+'.log')).open('a', buffering=1)
            command = [PYTHON, str(ROOT/'analysis/evaluate_distill_multiseed_20261029.py'),
                       '--variant',model,'--task',task,'--seeds',*map(str,SEEDS),
                       '--episodes','300','--output',str(folder)]
            env = dict(os.environ, MPLCONFIGDIR='/tmp/catch-mpl', XDG_CACHE_HOME='/tmp/catch-eval-cache',
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            proc = subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            active[name]=(proc,log)
            print('STARTED',name,'pid',proc.pid,flush=True)
            time.sleep(3)
        for name,(proc,log) in list(active.items()):
            code=proc.poll()
            if code is not None:
                log.close()
                del active[name]
                if code:
                    (OUT/'FAILED.json').write_text(json.dumps(dict(case=name,returncode=code),indent=2))
                    for child,_ in active.values():
                        child.terminate()
                    raise RuntimeError(name+' exited '+str(code)+'; see its .log')
                completed.append(name)
                print('FINISHED',name,flush=True)
        (OUT/'RUNNING.json').write_text(json.dumps(dict(seconds=time.monotonic()-start,
            active={name:proc.pid for name,(proc,_) in active.items()},completed=completed,
            remaining=[m+'_'+t for m,t in queue]),indent=2))
        if queue or active:
            time.sleep(5)
    (OUT/'EVALUATION_COMPLETE').write_text('All six formal evaluations completed\n')
    print('ALL EVALUATIONS COMPLETE',round(time.monotonic()-start,1),'seconds',flush=True)


if __name__=='__main__':
    main()
