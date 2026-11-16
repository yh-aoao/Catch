from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,os,subprocess
ROOT=Path('/home/isee/Catch/catch_it_copy');OUT=Path(__file__).resolve().parent;BASE=ROOT.parent/'analysis_artifacts/episode_dr_catch_20261116/results'
def run(job):
 name,obj=job;cfg=BASE/'configs'/(name+'.yaml');r=json.loads((BASE/name/obj/'request.json').read_text());r['name']=name+'_nominal_robot_range0';folder=OUT/r['name']/obj;folder.mkdir(parents=True,exist_ok=True);r['case_dir']=str(folder);(folder/'request.json').write_text(json.dumps(r,indent=2))
 env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl',PROBE_DR_SCALE='0',MULTITARGET_EVAL_REQUEST=str(folder/'request.json'))
 cmd=['/home/isee/miniconda3/envs/dcmm/bin/python',str(OUT/'worker_nominal_robot.py'),'--config-path',str(cfg.parent),'--config-name',cfg.stem,'hydra.job.chdir=false','test_object='+obj,'output_name=diagnose_nominal_robot_1116_'+name+'_'+obj]
 with (folder/'worker.log').open('w') as log:p=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
 assert p.returncode==0,str(folder/'worker.log');d=json.loads((folder/'summary.json').read_text());return {k:d[k] for k in ('name','object','successes','episodes')}
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=2) as pool:
  for r in pool.map(run,[('roll_best_236_00','apple'),('roll_best_236_00','lemon'),('bounce_best_1_57','original'),('bounce_best_1_57','baseball')]):print(json.dumps(r),flush=True)
 (OUT/'NOMINAL_ROBOT_COMPLETE').write_text('4 cases, 80 episodes\n')
