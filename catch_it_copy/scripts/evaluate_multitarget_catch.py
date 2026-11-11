"""Headless multi-target Catch evaluation. Run with the dcmm Python."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST=ROOT/'scripts/eval_multitarget_models_1111.json'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_reports(output,results):
    rows=sorted(results,key=lambda x:(x['name'],x['object']))
    fields=['name','motion','physics','object','episodes','successes','success_rate','stable_hold_episodes','stable_hold_rate']
    with (output/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    (output/'summary.json').write_text(json.dumps(rows,indent=2))
    totals=[]
    for name in sorted({r['name'] for r in rows}):
        a=[r for r in rows if r['name']==name];n=sum(r['episodes'] for r in a);k=sum(r['successes'] for r in a)
        h=sum(r['stable_hold_episodes'] or 0 for r in a)
        totals.append(dict(name=name,motion=a[0]['motion'],physics=a[0]['physics'],targets_completed=len(a),
            episodes=n,successes=k,success_rate=k/n,stable_hold_rate=h/n if a[0]['motion']=='roll' else None))
    (output/'model_totals.json').write_text(json.dumps(totals,indent=2))
    lines=['# Multi-target Catch evaluation','',
        'Roll: end-of-episode region retention; stable_hold is ever stable during the episode, not necessarily at the end.',
        'Bounce: controlled_hold_v1 continuous hold. Fixed and DR columns have different test distributions.',
        'Deterministic policy; explicit per-episode launch seeds; DR v1 parameters fixed per instance; dr_episode parameters resampled each episode.',
        'Per-object Wilson intervals and full physics samples are in the JSON results. No failed episodes are filtered.',
        '', '|Model|Mode|Target|Success|Rate|Roll ever stable|','|---|---|---|---:|---:|---:|']
    for r in rows:
        stable=f"{r['stable_hold_rate']:.1%}" if r['stable_hold_rate'] is not None else '—'
        lines.append(f"|{r['name']}|{r['physics']}|{r['object']}|{r['successes']}/{r['episodes']}|{r['success_rate']:.1%}|{stable}|")
    lines+=['','Totals below include completed targets only; check COMPLETE before interpreting a full run.','',
        '|Model|Targets completed|Success|Rate|','|---|---:|---:|---:|']
    for r in totals:lines.append(f"|{r['name']}|{r['targets_completed']}|{r['successes']}/{r['episodes']}|{r['success_rate']:.1%}|")
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,default=ROOT.parent/'11-11')
    p.add_argument('--models-file',type=Path,default=DEFAULT_MANIFEST)
    p.add_argument('--only',nargs='+',help='Subset of model names in the manifest')
    p.add_argument('--episodes',type=int,default=200,help='Completed episodes per model and target')
    p.add_argument('--instances',type=int,default=20,help='Physical samples per DR target; also vector environment count')
    p.add_argument('--jobs',type=int,default=2,help='Concurrent model/target jobs; each uses instances synchronous envs')
    p.add_argument('--seed',type=int,default=41000)
    p.add_argument('--objects',nargs='+',help='Optional target subset, primarily for smoke tests')
    p.add_argument('--catalog',type=Path,default=ROOT/'ycb_objects/calibration_data/object_benchmark_all_targets_v1.json')
    p.add_argument('--output',type=Path,default=None)
    p.add_argument('--resume',action='store_true',help='Skip complete cases only, validating the original run settings')
    a=p.parse_args()
    if min(a.episodes,a.instances,a.jobs)<1 or a.seed<0 or a.seed+a.instances>=2**32 or a.episodes%a.instances:
        p.error('Use positive counts, episodes divisible by instances, and a nonnegative 32-bit seed')
    models=json.loads(a.models_file.read_text());model_dir=a.model_dir.resolve()
    if not isinstance(models,list) or not models:p.error('models-file must contain a nonempty JSON list')
    names=[m['name'] for m in models]
    if len(set(names))!=len(names):p.error('Model names must be unique')
    if a.only and set(a.only)-set(names):p.error('Unknown model name in --only')
    models=[m.copy() for m in models if not a.only or m['name'] in a.only]
    for m in models:
        if not re.fullmatch(r'[A-Za-z0-9_-]+',m['name']) or m['motion'] not in ('roll','bounce') or m['physics'] not in ('fixed','dr','dr_episode'):
            p.error('Invalid manifest name/motion/physics')
        checkpoint=Path(m['checkpoint']);checkpoint=checkpoint if checkpoint.is_absolute() else model_dir/checkpoint
        if not checkpoint.is_file():p.error('Missing checkpoint: '+str(checkpoint))
        m['checkpoint']=str(checkpoint.resolve());m['checkpoint_sha256']=sha(checkpoint)
    data=json.loads(a.catalog.read_text())
    known=set(data['roll'])|set(data['bounce'])
    if a.objects and set(a.objects)-known:p.error('Unknown target in --objects')
    for m in models:
        if a.objects and not set(a.objects)&set(data[m['motion']]):p.error('No selected target for '+m['name'])
    source_names=['scripts/evaluate_multitarget_catch.py','scripts/_eval_multitarget_worker.py','train_DCMM.py',
        'configs/config.yaml','configs/train/DcmmPPO.yaml','gym_dcmm/envs/object_benchmark.py',
        'gym_dcmm/envs/roll_physical.py','gym_dcmm/envs/bounce_physical_v2.py',
        'ycb_objects/benchmark_geometry.py','ycb_objects/benchmark_randomization.py','ycb_objects/episode_randomization.py',
        'ycb_objects/calibration_data/balls.roll_table_priors_v1.json','ycb_objects/calibration_data/balls.three_ball_training_v1.json',
        'gym_dcmm/envs/DcmmVecEnv_roll_645edc4.py','gym_dcmm/envs/bounce_grasp.py',
        'gym_dcmm/utils/roll_evaluation.py','gym_dcmm/algs/ppo_dcmm_645edc4/ppo_dcmm_catch_two_stage.py',
        'gym_dcmm/algs/ppo_dcmm_5fe75d5f/ppo_dcmm_catch_hold.py']
    spec=dict(episodes=a.episodes,instances=a.instances,seed=a.seed,models=models,objects=a.objects,
        catalog_sha256=sha(a.catalog),source_sha256={n:sha(ROOT/n) for n in source_names})
    output=(a.output or ROOT/'evaluation_results'/datetime.now().strftime('multitarget_%Y%m%d_%H%M%S_%f')).resolve()
    if a.resume:
        if not (output/'run_spec.json').is_file() or json.loads((output/'run_spec.json').read_text())!=spec:
            p.error('Resume settings/files differ; use original arguments or a new output directory')
        if sha(output/'catalog_used.json')!=spec['catalog_sha256']:p.error('Saved catalog was modified')
    else:
        if output.exists() and any(output.iterdir()):p.error('Output is not empty; use --resume or a new directory')
        output.mkdir(parents=True,exist_ok=True)
        (output/'run_spec.json').write_text(json.dumps(spec,indent=2));shutil.copyfile(a.catalog,output/'catalog_used.json')
    env=dict(os.environ,PATH=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH',''),
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL=os.environ.get('MUJOCO_GL','egl'))
    cases=[]
    for m in models:
        # Resolve the exact production training/viewer options, with test=True.
        args=['bash',str(ROOT/'scripts/train_object_benchmark.sh'),m['motion'],
            'task=Catching_TwoStage','test=True','viewer=false','imshow_cam=false','device_id=-1',
            'hydra.job.chdir=false','vector_mode=sync',f'num_envs={a.instances}',f'train.ppo.num_actors={a.instances}',
            'checkpoint_tracking=','checkpoint_catching='+m['checkpoint'],'benchmark_catalog='+str(output/'catalog_used.json'),
            '+benchmark_options.domain_randomization='+({'dr':'physical_dr_v1','dr_episode':'physical_dr_episode_v1'}.get(m['physics'],'off')),
            m['motion']+'_ball_mode=fixed','seed='+str(a.seed)]
        config_dir=output/'configs';config_dir.mkdir(exist_ok=True)
        config=config_dir/(m['name']+'.yaml')
        if not a.resume:
            resolved=subprocess.run(args+['--cfg','job','--resolve'],cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
            config.write_text(resolved.stdout)
        if not config.is_file():p.error('Missing saved config: '+str(config))
        for obj in data[m['motion']]:
            if a.objects and obj not in a.objects:continue
            folder=output/m['name']/obj;folder.mkdir(parents=True,exist_ok=True)
            request=dict(m,object=obj,seed=a.seed,episodes=a.episodes,instances=a.instances,case_dir=str(folder))
            cases.append((request,config))
    def run(case):
        request,config=case;folder=Path(request['case_dir']);result=folder/'summary.json'
        if a.resume and result.exists():
            d=json.loads(result.read_text())
            if d.get('completed') and d['episodes']==a.episodes and d['checkpoint_sha256']==request['checkpoint_sha256']:return d
        (folder/'request.json').write_text(json.dumps(request,indent=2))
        e=dict(env,MULTITARGET_EVAL_REQUEST=str(folder/'request.json'))
        command=[sys.executable,str(ROOT/'scripts/_eval_multitarget_worker.py'),'--config-path',str(config.parent),
            '--config-name',config.stem,'test_object='+request['object'],
            'output_name=eval_'+output.name+'_'+request['name']+'_'+request['object'],'hydra.job.chdir=false']
        with (folder/'worker.log').open('w') as log:
            status=subprocess.run(command,cwd=ROOT,env=e,stdout=log,stderr=subprocess.STDOUT)
        if status.returncode or not result.exists():raise RuntimeError('Evaluation failed; see '+str(folder/'worker.log'))
        return json.loads(result.read_text())
    print(f'Output: {output}\nCases: {len(cases)}; counted episodes: {len(cases)*a.episodes}; no training',flush=True)
    results=[];failures=[]
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        pending={pool.submit(run,c):c[0] for c in cases}
        for f in as_completed(pending):
            try:
                d=f.result();results.append(d);print(f"{d['name']} {d['object']}: {d['successes']}/{d['episodes']} = {d['success_rate']:.1%}",flush=True)
                write_reports(output,results)
            except Exception as error:
                failures.append(dict(case=pending[f],error=str(error)));print(str(error),file=sys.stderr,flush=True)
    (output/'failures.json').write_text(json.dumps(failures,indent=2))
    if failures:return 1
    for name,digest in spec['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Source changed during evaluation: '+name)
    for m in models:
        if sha(m['checkpoint'])!=m['checkpoint_sha256']:raise RuntimeError('Checkpoint changed: '+m['checkpoint'])
    (output/'COMPLETE').write_text(f'{len(cases)} cases x {a.episodes} complete episodes\n')
    print('Complete. Summary: '+str(output/'REPORT.md'),flush=True)
    return 0

if __name__=='__main__':sys.exit(main())
