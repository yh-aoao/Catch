import concurrent.futures,subprocess,pathlib,os,json
root=pathlib.Path('/home/isee/Catch');out=root/'analysis_artifacts/catch_three_ball_20261105'
env=dict(os.environ,PYTHONPATH=str(root/'catch_it_copy'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl')
models={'roll':'roll_catch_best_reward_270.37.pth','bounce':'bounce_catch_best_reward_13.68.pth'}
def run(job):
 task,ball=job;args=['/home/isee/miniconda3/envs/dcmm/bin/python',str(root/'analysis/evaluate_catch_balls_20261102.py'),'--task',task,'--object',ball,'--checkpoint',str(root/'11-5'/models[task]),'--config',str(out/f'{task}_config_reconstructed.yaml'),'--output',str(out/f'{task}_{ball}'),'--episodes','100','--seed','0']
 with (out/f'{task}_{ball}.log').open('w') as f:p=subprocess.run(args,cwd=root/'catch_it_copy',env=env,stdout=f,stderr=subprocess.STDOUT)
 print(task,ball,p.returncode,flush=True)
 if p.returncode:print((out/f'{task}_{ball}.log').read_text()[-2500:],flush=True)
 return dict(task=task,object=ball,exit_code=p.returncode)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(run,[(m,b) for m in models for b in ['original','tennis_ball','baseball']]))
(out/'run_status.json').write_text(json.dumps(results,indent=2))
