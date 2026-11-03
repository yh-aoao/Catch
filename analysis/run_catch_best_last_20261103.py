"""Paired best/last evaluation; no training or source mutations."""
import subprocess,os,json,concurrent.futures,hashlib,shutil
from pathlib import Path
root=Path('/home/isee/Catch');out=root/'analysis_artifacts/catch_best_last_20261103'
from omegaconf import OmegaConf
models={'roll_best':'roll_catch_best_reward_262.02.pth','roll_last':'roll_catch_last.pth','bounce_best':'bounce_catch_best_reward_8.55.pth','bounce_last':'bounce_catch_last.pth'}
for task in ('roll','bounce'):
 cfg=OmegaConf.load(root/'analysis_artifacts/catch_three_ball_20261102'/(task+'_config_reconstructed.yaml'))
 if task=='bounce':cfg.bounce_grasp_training={'track_lr_scale':1.0,'freeze_track_steps':0,'freeze_track_normalization':False}
 OmegaConf.save(cfg,out/(task+'_config_reconstructed.yaml'))
meta={'episodes_per_condition':100,'seeds':[0,99],'configuration':'Reconstructed from prior equal-v2 commands; original server resolved_config.yaml absent. Bounce optimization schedule confirmed in both checkpoints.','models':{k:{'path':str(root/'11-3'/v),'sha256':hashlib.sha256((root/'11-3'/v).read_bytes()).hexdigest()} for k,v in models.items()}}
(out/'evaluation_notes.json').write_text(json.dumps(meta,indent=2))
shutil.copy2(root/'analysis/evaluate_catch_balls_20261102.py',out/'evaluator_snapshot.py')
env=dict(os.environ,PYTHONPATH=str(root/'catch_it_copy'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl',MPLCONFIGDIR='/tmp/catch_eval_mpl_20261103')
def run(item):
 key,ball=item;task=key.split('_')[0]
 cmd=['/home/isee/miniconda3/envs/dcmm/bin/python',str(root/'analysis/evaluate_catch_balls_20261102.py'),'--task',task,'--object',ball,'--checkpoint',str(root/'11-3'/models[key]),'--config',str(out/(task+'_config_reconstructed.yaml')),'--output',str(out/(key+'_'+ball)),'--episodes','100']
 with (out/(key+'_'+ball+'.log')).open('w') as f:p=subprocess.run(cmd,cwd=root/'catch_it_copy',env=env,stdout=f,stderr=subprocess.STDOUT)
 result={'model':key,'ball':ball,'exit':p.returncode}
 print(json.dumps(result),flush=True);return result
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
 results=list(pool.map(run,[(k,b) for k in models for b in ('original','tennis_ball','baseball')]))
(out/'run_status.json').write_text(json.dumps(results,indent=2))
