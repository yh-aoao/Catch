import subprocess,concurrent.futures,os,pathlib,json
root=pathlib.Path('/home/isee/Catch/catch_it_copy');out=root.parent/'analysis_artifacts/three_ball_catch_warm_start_20261109'
env=dict(os.environ,PATH='/home/isee/miniconda3/envs/dcmm/bin:'+os.environ['PATH'],OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MUJOCO_GL='egl',WANDB_MODE='disabled')
def run(job):
 m,v=job;tag=f'{m}_{v}'
 cfg=out/f'{tag}_resolved.yaml'
 with cfg.open('w') as f:subprocess.run(['bash','scripts/finetune_three_ball_catch_parameter_v1.sh',m,v,'device_id=-1','hydra.job.chdir=false','--cfg','job','--resolve'],cwd=root,env=env,stdout=f,stderr=subprocess.PIPE,check=True)
 args=['/home/isee/miniconda3/envs/dcmm/bin/python','analysis/check_catch_warm_start_20261109.py','--config-path',str(out),'--config-name',f'{tag}_resolved','hydra.job.chdir=false','num_envs=3','train.ppo.num_actors=3','train.ppo.horizon_length=8','train.ppo.minibatch_size=12','train.ppo.mini_epochs=1','train.ppo.max_agent_steps=72','wandb_mode=disabled',f'output_name=catch_warm_smoke_{tag}_20261109']
 with (out/f'{tag}_smoke.log').open('w') as f:p=subprocess.run(args,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
 print(tag,p.returncode,flush=True)
 if p.returncode:print((out/f'{tag}_smoke.log').read_text()[-2500:],flush=True)
 return dict(motion=m,variant=v,exit_code=p.returncode)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(run,[(m,v) for m in ['roll','bounce'] for v in ['dr','fixed']]))
(out/'smoke_results.json').write_text(json.dumps(results,indent=2))
