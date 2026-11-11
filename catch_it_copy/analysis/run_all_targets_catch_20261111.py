"""Evaluate user-labelled fixed and DR models against current launch profiles."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parent/'analysis_artifacts/all_targets_catch_20261111'
OBJECTS={'roll':['original','tennis_ball','baseball','orange','apple','lemon','tomato_soup_can'],
         'bounce':['original','tennis_ball','baseball','racquetball','golf_ball','softball']}
MODELS=[('roll','261.94','fixed'),('roll','229.39','dr'),('roll','222.64','dr'),
        ('bounce','3.39','fixed'),('bounce','3.96','dr')]

def run(job):
    motion,reward,mode,obj=job
    tag=motion+'_'+reward+'_'+mode;folder=OUT/tag;folder.mkdir(exist_ok=True)
    checkpoint=ROOT.parent/'11-11'/f'{motion}_catch_best_reward_{reward}.pth'
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl',
        EVAL_MOTION=motion,EVAL_OBJECT=obj,EVAL_CHECKPOINT=str(checkpoint),EVAL_RESULT=str(folder/f'{motion}_{obj}.json'))
    entry='analysis/quick_dr_catch_eval_20261109.py' if mode=='dr' else 'analysis/quick_fixed_catch_eval_20261109.py'
    args=['/home/isee/miniconda3/envs/dcmm/bin/python',entry,'--config-path',str(OUT),'--config-name',motion+'_resolved',
        'benchmark_catalog='+str(OUT/'catalog_used.json'),'hydra.job.chdir=false','test=True','task=Catching_TwoStage',
        'num_envs=4','train.ppo.num_actors=4','seed=41000','+benchmark_options.domain_randomization='+('physical_dr_v1' if mode=='dr' else 'off'),
        'vector_mode=sync','device_id=-1','viewer=false','imshow_cam=false',motion+'_ball_mode=fixed',
        'test_object='+obj,'checkpoint_tracking=','checkpoint_catching='+str(checkpoint),'output_name=quick_1111_'+tag+'_'+obj]
    with (folder/f'{motion}_{obj}.log').open('w') as log:
        p=subprocess.run(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    result=dict(motion=motion,reward=reward,mode=mode,object=obj,exit_code=p.returncode)
    if p.returncode==0:
        d=json.loads((folder/f'{motion}_{obj}.json').read_text());result['successes']=d['successes']
    else:print((folder/f'{motion}_{obj}.log').read_text()[-2500:],flush=True)
    print(json.dumps(result),flush=True)
    return result

if __name__=='__main__':
    # Interleave models within each target to expose failures early.
    jobs=[(m,r,v,o) for m in OBJECTS for o in OBJECTS[m] for mm,r,v in MODELS if mm==m]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(run,jobs))
    (OUT/'run_status.json').write_text(json.dumps(results,indent=2))
