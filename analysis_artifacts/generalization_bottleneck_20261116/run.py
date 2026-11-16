from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,os,subprocess,hashlib
ROOT=Path('/home/isee/Catch/catch_it_copy');OUT=Path(__file__).resolve().parent
BASE=ROOT.parent/'analysis_artifacts/episode_dr_catch_20261116/results';PYTHON='/home/isee/miniconda3/envs/dcmm/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl')
 cases=[]
 for name,motion in [('roll_best_236_00','roll'),('bounce_best_1_57','bounce')]:
  cfg=BASE/'configs'/(name+'.yaml')
  targets=json.loads((BASE/'catalog_used.json').read_text())[motion]
  for obj in targets:
   base=json.loads((BASE/name/obj/'request.json').read_text())
   for scale in (.5,0.):
    request=base.copy();request['name']=name+'_range'+str(scale);folder=OUT/request['name']/obj;folder.mkdir(parents=True,exist_ok=True);request['case_dir']=str(folder)
    (folder/'request.json').write_text(json.dumps(request,indent=2));cases.append((request,cfg,scale))
 # Full-range micro-contact diagnostics on the two hardest old-ball Bounce cases.
 for obj in ('original','baseball'):
  base=json.loads((BASE/'bounce_best_1_57'/obj/'request.json').read_text());base['name']='bounce_best_1_57_range1.0';folder=OUT/base['name']/obj;folder.mkdir(parents=True,exist_ok=True);base['case_dir']=str(folder);(folder/'request.json').write_text(json.dumps(base,indent=2));cases.append((base,BASE/'configs/bounce_best_1_57.yaml',1.))
 # Specialist reference under the current per-episode DR test, not legacy physics.
 ckpt=ROOT.parent/'11-5/bounce_catch_best_reward_13.68.pth'
 for obj in ('original','tennis_ball','baseball'):
  base=json.loads((BASE/'bounce_best_1_57'/obj/'request.json').read_text());base.update(name='bounce_three_ball_13_68',checkpoint=str(ckpt),checkpoint_sha256=sha(ckpt));folder=OUT/base['name']/obj;folder.mkdir(parents=True,exist_ok=True);base['case_dir']=str(folder);(folder/'request.json').write_text(json.dumps(base,indent=2));cases.append((base,BASE/'configs/bounce_best_1_57.yaml',1.))
 sources=['ycb_objects/episode_randomization.py','ycb_objects/benchmark_randomization.py','gym_dcmm/envs/object_benchmark.py','gym_dcmm/envs/bounce_grasp.py','scripts/_eval_multitarget_worker.py','gym_dcmm/utils/roll_evaluation.py']
 spec=dict(episodes_per_case=20,instances=4,source_sha256={s:sha(ROOT/s) for s in sources},baseline_source=str(BASE),cases=[dict(request=r,config=str(c),scale=s) for r,c,s in cases]);(OUT/'run_spec.json').write_text(json.dumps(spec,indent=2))
 def run(case):
  r,cfg,scale=case;folder=Path(r['case_dir']);e=dict(env,PROBE_DR_SCALE=str(scale),MULTITARGET_EVAL_REQUEST=str(folder/'request.json'))
  cmd=[PYTHON,str(OUT/'worker.py'),'--config-path',str(cfg.parent),'--config-name',cfg.stem,'hydra.job.chdir=false','test_object='+r['object'],'checkpoint_catching='+r['checkpoint'],'output_name=diagnose_1116_'+r['name']+'_'+r['object']]
  with (folder/'worker.log').open('w') as log:p=subprocess.run(cmd,cwd=ROOT,env=e,stdout=log,stderr=subprocess.STDOUT)
  if p.returncode:raise RuntimeError(str(folder/'worker.log'))
  d=json.loads((folder/'summary.json').read_text());return d
 print('Cases',len(cases),'episodes',len(cases)*20,flush=True)
 results=[];errors=[]
 with ThreadPoolExecutor(max_workers=4) as pool:
  for f in as_completed([pool.submit(run,c) for c in cases]):
   try:
    d=f.result();results.append(d);print(d['name'],d['object'],d['successes'],'/20',flush=True);(OUT/'summary.json').write_text(json.dumps(results,indent=2))
   except Exception as error:errors.append(str(error));print('ERROR',str(error),flush=True)
 (OUT/'errors.json').write_text(json.dumps(errors,indent=2));assert not errors,errors
 for s,h in spec['source_sha256'].items():assert sha(ROOT/s)==h,s
 (OUT/'COMPLETE').write_text(str(len(cases))+' cases, '+str(len(cases)*20)+' completed episodes\n')
if __name__=='__main__':main()
