import concurrent.futures,subprocess,os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'analysis_artifacts/object_benchmark_20261103'
MODELS={'roll':'roll_catch_best_reward_269.97.pth','bounce':'bounce_catch_best_reward_9.40.pth'}
def run(item):
 task,ball=item;output=OUT/'baseline_regression'/(task+'_'+ball)
 env=dict(os.environ,PYTHONPATH=str(ROOT/'catch_it_copy'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR='/tmp/benchmark_baseline_mpl_'+task+ball)
 output.parent.mkdir(exist_ok=True)
 with (output.parent/(task+'_'+ball+'.log')).open('w') as f:
  p=subprocess.run(['/home/isee/miniconda3/envs/dcmm/bin/python',str(ROOT/'analysis/evaluate_catch_balls_20261102.py'),'--task',task,'--object',ball,'--config',str(ROOT/'analysis_artifacts/catch_three_ball_20261103_v2'/(task+'_config_reconstructed.yaml')),'--checkpoint',str(ROOT/'11-3-2'/MODELS[task]),'--output',str(output),'--episodes','1'],cwd=ROOT/'catch_it_copy',env=env,stdout=f,stderr=subprocess.STDOUT)
 if p.returncode:return dict(task=task,ball=ball,exit=p.returncode)
 a=json.loads((output/'episodes.jsonl').read_text().splitlines()[0]);b=json.loads((ROOT/'analysis_artifacts/catch_three_ball_20261103_v2'/(task+'_best_'+ball)/'episodes.jsonl').read_text().splitlines()[0])
 checks={k:a[k]==b[k] for k in ('initial','final','reward','success')}
 return dict(task=task,ball=ball,exit=0,checks=checks)
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:results=list(pool.map(run,[(m,b) for m in ('roll','bounce') for b in ('original','tennis_ball','baseball')]))
(OUT/'baseline_regression.json').write_text(json.dumps(results,indent=2));print(results)
assert all(x['exit']==0 and all(x['checks'].values()) for x in results)
