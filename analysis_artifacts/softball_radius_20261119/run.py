from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,subprocess,os,hashlib,sys
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]/'catch_it_copy'
BASE=OUT.parent/'catch_creation_1119_quick'
request=json.loads((BASE/'bounce_best_1119/softball/request.json').read_text());config=BASE/'configs/bounce_best_1119.yaml'
paths=['gym_dcmm/envs/object_benchmark.py','gym_dcmm/envs/bounce_grasp.py','ycb_objects/benchmark_geometry.py','scripts/_eval_multitarget_worker.py']
hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in paths}
(OUT/'run_spec.json').write_text(json.dumps(dict(scales=[1.,.95,.9],episodes_per_scale=20,source_sha256=hashes,checkpoint_sha256=request['checkpoint_sha256']),indent=2))
def run(scale):
 folder=OUT/('r'+str(round(scale*100)));folder.mkdir(exist_ok=False)
 r=dict(request,name=folder.name,case_dir=str(folder),softball_radius_scale=scale)
 (folder/'request.json').write_text(json.dumps(r,indent=2))
 env=dict(os.environ,MUJOCO_GL='egl',MPLCONFIGDIR='/tmp/catch-matplotlib',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',SOFTBALL_RADIUS_SCALE=str(scale),MULTITARGET_EVAL_REQUEST=str(folder/'request.json'))
 cmd=[sys.executable,str(OUT/'worker.py'),'--config-path',str(config.parent),'--config-name',config.stem,'test_object=softball','hydra.job.chdir=false','output_name=softball_radius_probe_'+folder.name]
 with (folder/'worker.log').open('w') as stream:result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
 assert result.returncode==0,str(folder/'worker.log')
 data=json.loads((folder/'summary.json').read_text());print(folder.name,data['successes'],'/20',flush=True);return data
results=[]
with ThreadPoolExecutor(max_workers=3) as pool:
 for future in as_completed([pool.submit(run,s) for s in [1.,.95,.9]]):
  results.append(future.result());(OUT/'summary.json').write_text(json.dumps(results,indent=2))
for f,h in hashes.items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h,f
(OUT/'COMPLETE').write_text('3 conditions x20 episodes\n')
