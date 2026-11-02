import subprocess,os,json,concurrent.futures
from pathlib import Path
root=Path('/home/isee/Catch'); out=root/'analysis_artifacts/catch_three_ball_20261102'
env=dict(os.environ,PYTHONPATH=str(root/'catch_it_copy'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl')
def run(task,ball):
 checkpoint=root/'11-2'/({'roll':'roll_catch_best_reward_250.65.pth','bounce':'bounce_catch_best_reward_-0.73.pth'}[task])
 cmd=['/home/isee/miniconda3/envs/dcmm/bin/python',str(root/'analysis/evaluate_catch_balls_20261102.py'),'--task',task,'--object',ball,'--checkpoint',str(checkpoint),'--config',str(out/(task+'_config_reconstructed.yaml')),'--output',str(out/(task+'_'+ball)),'--episodes','100']
 with (out/(task+'_'+ball+'.log')).open('w') as f:
  p=subprocess.run(cmd,cwd=root/'catch_it_copy',env=env,stdout=f,stderr=subprocess.STDOUT)
 return dict(task=task,ball=ball,exit=p.returncode)
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
 results=list(pool.map(lambda x:run(*x),[(t,b) for t in ('roll','bounce') for b in ('original','tennis_ball','baseball')]))
(out/'run_status.json').write_text(json.dumps(results,indent=2))
print(results)
