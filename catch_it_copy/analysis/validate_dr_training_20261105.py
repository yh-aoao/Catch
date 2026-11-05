import subprocess,concurrent.futures,os,pathlib,json
root=pathlib.Path('/home/isee/Catch/catch_it_copy');out=root.parent/'analysis_artifacts/physical_dr_v1_20261105'
env=dict(os.environ,PATH='/home/isee/miniconda3/envs/dcmm/bin:'+os.environ['PATH'],OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MUJOCO_GL='egl',WANDB_MODE='disabled')
def run(job):
 motion,stage=job;n=7 if motion=='roll' else 6
 args=['bash','scripts/train_all_targets_dr_v1.sh',motion,stage,'device_id=-1','hydra.job.chdir=false',f'num_envs={n}','train.ppo.horizon_length=8','train.ppo.mini_epochs=1',f'train.ppo.max_agent_steps={n*24}','wandb_mode=disabled',f'output_name=dr_smoke_{motion}_{stage}_20261105']
 if stage=='catch':
  folder='roll_seven_object_tracking_v1_seed123' if motion=='roll' else 'bounce_six_ball_tracking_v1_seed123'
  file='roll_track_best_reward_24.83.pth' if motion=='roll' else 'bounce_track_best_reward_19.08.pth'
  args.append('checkpoint_tracking='+str(root/'outputs'/folder/'2026-11-04/01:33:28/nn'/file))
 with (out/f'{motion}_{stage}_smoke.log').open('w') as f:p=subprocess.run(args,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
 print(motion,stage,p.returncode,flush=True)
 if p.returncode:print((out/f'{motion}_{stage}_smoke.log').read_text()[-1800:],flush=True)
 return dict(motion=motion,stage=stage,exit_code=p.returncode)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:results=list(ex.map(run,[(m,s) for m in ['roll','bounce'] for s in ['track','catch']]))
(out/'training_smoke.json').write_text(json.dumps(results,indent=2))
