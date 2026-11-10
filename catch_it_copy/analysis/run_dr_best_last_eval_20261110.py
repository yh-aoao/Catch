"""Same DR physical samples for multi-target best/last Catch comparison."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parent/'analysis_artifacts/all_targets_dr_best_last_20261110'
OBJECTS={'roll':['original','tennis_ball','baseball','orange','apple','lemon','tomato_soup_can'],
         'bounce':['original','tennis_ball','baseball','racquetball','golf_ball','softball']}
BEST={'roll':'roll_catch_best_reward_237.70.pth','bounce':'bounce_catch_best_reward_-1.27.pth'}

def run(job):
    motion,variant,obj=job
    folder=OUT/variant;folder.mkdir(exist_ok=True)
    checkpoint=ROOT.parent/'11-10'/(BEST[motion] if variant=='best' else motion+'_catch_last.pth')
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl',
        EVAL_MOTION=motion,EVAL_OBJECT=obj,EVAL_CHECKPOINT=str(checkpoint),EVAL_RESULT=str(folder/f'{motion}_{obj}.json'))
    args=['/home/isee/miniconda3/envs/dcmm/bin/python','analysis/quick_dr_catch_eval_20261109.py',
        '--config-path',str(ROOT.parent/'analysis_artifacts/all_targets_catch_setup_20261104'),
        '--config-name',motion+'_resolved',
        'benchmark_catalog='+str(ROOT/'ycb_objects/calibration_data/object_benchmark_all_targets_v1.json'),
        'hydra.job.chdir=false','test=True','task=Catching_TwoStage','num_envs=4','train.ppo.num_actors=4',
        'seed=41000','+benchmark_options.domain_randomization=physical_dr_v1','vector_mode=sync',
        'device_id=-1','viewer=false','imshow_cam=false',motion+'_ball_mode=fixed','test_object='+obj,
        'checkpoint_tracking=','checkpoint_catching='+str(checkpoint),
        'output_name=quick_dr_1110_'+motion+'_'+variant+'_'+obj]
    with (folder/f'{motion}_{obj}.log').open('w') as log:
        p=subprocess.run(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    result=dict(motion=motion,variant=variant,object=obj,exit_code=p.returncode)
    if p.returncode==0:
        d=json.loads((folder/f'{motion}_{obj}.json').read_text());result['successes']=d['successes']
    else:print((folder/f'{motion}_{obj}.log').read_text()[-2000:],flush=True)
    print(json.dumps(result),flush=True)
    return result

if __name__=='__main__':
    OUT.mkdir(exist_ok=True,parents=True)
    jobs=[(m,v,o) for m in OBJECTS for o in OBJECTS[m] for v in ['best','last']]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(run,jobs))
    (OUT/'run_status.json').write_text(json.dumps(results,indent=2))
