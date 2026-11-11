from pathlib import Path
import json,os,subprocess,hashlib,sys
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path('/home/isee/Catch/catch_it_copy');OUT=Path(__file__).resolve().parent
PYTHON='/home/isee/miniconda3/envs/dcmm/bin/python'
jobs=[]
for motion,obj,episodes in [('bounce','original',60),('bounce','tennis_ball',60),('roll','apple',40),('roll','lemon',40)]:
 for scale in (1.,.5,0.):
  name=f'{motion}_{obj}_range{scale}';folder=OUT/name;folder.mkdir(exist_ok=True)
  config=(ROOT/'evaluation_results/bounce_484_last_200_new/configs/bounce_dr_best_4_84.yaml') if motion=='bounce' else ROOT/'evaluation_results/roll_23161_200/configs/roll_dr_231_61.yaml'
  checkpoint=Path('/home/isee/Catch/11-11')/('bounce_catch_best_reward_4.84.pth' if motion=='bounce' else 'roll_catch_best_reward_231.61.pth')
  req=dict(name=name,motion=motion,object=obj,physics='dr',checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),instances=10,episodes=episodes,seed=41000,case_dir=str(folder))
  (folder/'request.json').write_text(json.dumps(req,indent=2));jobs.append((req,config,scale))
def run(job):
 req,cfg,scale=job;folder=Path(req['case_dir'])
 env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl',MULTITARGET_EVAL_REQUEST=str(folder/'request.json'),PROBE_DR_SCALE=str(scale))
 cmd=[PYTHON,str(OUT/'probe_worker.py'),'--config-path',str(cfg.parent),'--config-name',cfg.stem,'test_object='+req['object'],'num_envs=10','train.ppo.num_actors=10','hydra.job.chdir=false','output_name=diagnosis_'+req['name']]
 with (folder/'worker.log').open('w') as log:p=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
 if p.returncode:raise RuntimeError(req['name']+' failed: '+str(folder/'worker.log'))
 result=json.loads((folder/'summary.json').read_text());return {k:result[k] for k in ('name','episodes','successes','success_rate','reasons')}
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=4) as pool:
  for f in as_completed([pool.submit(run,j) for j in jobs]):print(json.dumps(f.result()),flush=True)
 (OUT/'PROBES_COMPLETE').write_text('12 cases, 600 episodes\n')
