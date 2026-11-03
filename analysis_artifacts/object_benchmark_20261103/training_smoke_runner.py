import subprocess,concurrent.futures,os,json
from pathlib import Path
root=Path('/home/isee/Catch');out=root/'analysis_artifacts/object_benchmark_20261103'
def run(item):
 motion,task=item
 args=['bash','scripts/train_object_benchmark.sh',motion,'task='+task,'device_id=-1','num_envs=2','vector_mode=sync','train.ppo.horizon_length=8','train.ppo.minibatch_size=16','train.ppo.mini_epochs=1','train.ppo.max_agent_steps=64','output_name=benchmark_smoke_'+motion+'_'+task+'_20261103']
 if task=='Catching_TwoStage':args+=['checkpoint_tracking='+str(root/'11-2'/({'roll':'roll_track_best_reward_30.07.pth','bounce':'bounce_track_best_reward_20.04.pth'}[motion]))]
 env=dict(os.environ,PATH='/home/isee/miniconda3/envs/dcmm/bin:'+os.environ['PATH'],OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR='/tmp/benchmark_train_mpl_'+motion+'_'+task)
 with (out/('train_'+motion+'_'+task+'.log')).open('w') as f:r=subprocess.run(args,cwd=root/'catch_it_copy',env=env,stdout=f,stderr=subprocess.STDOUT)
 return dict(motion=motion,task=task,exit=r.returncode)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(run,[(m,t) for m in ('roll','bounce') for t in ('Tracking','Catching_TwoStage')]))
(out/'training_smoke.json').write_text(json.dumps(results,indent=2));print(results)
