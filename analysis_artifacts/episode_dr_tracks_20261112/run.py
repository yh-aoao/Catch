from pathlib import Path
import json,os,sys,hashlib,subprocess
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path('/home/isee/Catch/catch_it_copy');OUT=Path(__file__).resolve().parent
PYTHON='/home/isee/miniconda3/envs/dcmm/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run():
 catalog=ROOT/'ycb_objects/calibration_data/object_benchmark_all_targets_v1.json'
 (OUT/'catalog_used.json').write_bytes(catalog.read_bytes());data=json.loads(catalog.read_text())
 env=dict(os.environ,PATH=str(Path(PYTHON).parent)+os.pathsep+os.environ.get('PATH',''),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl')
 sources=['train_DCMM.py','gym_dcmm/envs/object_benchmark.py','ycb_objects/episode_randomization.py','ycb_objects/benchmark_randomization.py','scripts/_eval_multitarget_worker.py','gym_dcmm/algs/ppo_dcmm_645edc4/ppo_dcmm_track.py','gym_dcmm/algs/ppo_dcmm_july22/ppo_dcmm_track.py','gym_dcmm/utils/roll_evaluation.py']
 spec=dict(episodes=20,instances=4,seed=41000,physics='physical_dr_episode_v1',sources={s:sha(ROOT/s) for s in sources},catalog_sha256=sha(catalog),models={})
 jobs=[]
 for motion,filename in [('roll','roll_track_best_reward_24.52.pth'),('bounce','bounce_track_best_reward_19.56.pth')]:
  checkpoint=ROOT.parent/'11-12'/filename;digest=sha(checkpoint);spec['models'][motion]=dict(checkpoint=str(checkpoint),sha256=digest)
  cmd=['bash','scripts/train_all_targets_episode_dr_v1.sh',motion,'track','test=True','device_id=-1','viewer=false','imshow_cam=false','vector_mode=sync','num_envs=4','train.ppo.num_actors=4','checkpoint_tracking='+str(checkpoint),'checkpoint_catching=','seed=41000','benchmark_catalog='+str(OUT/'catalog_used.json'),'--cfg','job','--resolve']
  config=OUT/(motion+'.yaml');resolved=subprocess.run(cmd,cwd=ROOT,env=env,text=True,capture_output=True,check=True);config.write_text(resolved.stdout)
  for obj in data[motion]:
   folder=OUT/motion/obj;folder.mkdir(parents=True,exist_ok=False)
   req=dict(name=motion+'_track_1112',motion=motion,object=obj,physics='dr_episode',checkpoint=str(checkpoint),checkpoint_sha256=digest,seed=41000,episodes=20,instances=4,case_dir=str(folder))
   (folder/'request.json').write_text(json.dumps(req,indent=2));jobs.append((req,config))
 (OUT/'run_spec.json').write_text(json.dumps(spec,indent=2))
 def case(job):
  req,cfg=job;folder=Path(req['case_dir']);e=dict(env,MULTITARGET_EVAL_REQUEST=str(folder/'request.json'))
  cmd=[PYTHON,str(OUT/'worker.py'),'--config-path',str(OUT),'--config-name',cfg.stem,'hydra.job.chdir=false',req['motion']+'_ball_mode=fixed','test_object='+req['object'],'output_name=quick_episode_track_1112_'+req['motion']+'_'+req['object']]
  with (folder/'worker.log').open('w') as f:p=subprocess.run(cmd,cwd=ROOT,env=e,stdout=f,stderr=subprocess.STDOUT)
  if p.returncode:raise RuntimeError('Failure: '+str(folder/'worker.log'))
  result=json.loads((folder/'summary.json').read_text());return result
 results=[];errors=[]
 with ThreadPoolExecutor(max_workers=3) as pool:
  for f in as_completed([pool.submit(case,j) for j in jobs]):
   try:
    d=f.result();results.append(d);print(d['motion'],d['object'],d['successes'],'/',d['episodes'],flush=True)
    (OUT/'summary.json').write_text(json.dumps(results,indent=2))
   except Exception as e:errors.append(str(e));print(str(e),flush=True)
 (OUT/'errors.json').write_text(json.dumps(errors,indent=2))
 assert not errors,errors
 for name,h in spec['sources'].items():assert sha(ROOT/name)==h,name
 assert sha(catalog)==spec['catalog_sha256']
 for m,info in spec['models'].items():assert sha(info['checkpoint'])==info['sha256']
 (OUT/'COMPLETE').write_text('13 cases x 20 episodes, 260 total\n')
if __name__=='__main__':run()
