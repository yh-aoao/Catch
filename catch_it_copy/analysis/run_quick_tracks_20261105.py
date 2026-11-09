import concurrent.futures, subprocess,os,json,pathlib,time
root=pathlib.Path('/home/isee/Catch/catch_it_copy'); out=root.parent/'analysis_artifacts/all_targets_track_quick_20261105'
sets={'roll':('roll_seven_object_tracking_v1_seed123','roll_track_best_reward_26.25.pth',['original','tennis_ball','baseball','orange','apple','lemon','tomato_soup_can']), 'bounce':('bounce_six_ball_tracking_v1_seed123','bounce_track_best_reward_20.19.pth',['original','tennis_ball','baseball','racquetball','golf_ball','softball'])}
def run(job):
 motion,obj=job; folder,model,_=sets[motion]; config=root/'outputs'/folder/'2026-11-04/01:33:28'; checkpoint=root.parent/'11-5'/model
 env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl',EVAL_MOTION=motion,EVAL_OBJECT=obj,EVAL_CHECKPOINT=str(checkpoint),EVAL_RESULT=str(out/f'{motion}_{obj}.json'))
 args=[str(pathlib.Path('/home/isee/miniconda3/envs/dcmm/bin/python')),'analysis/quick_track_eval_20261104.py','--config-path',str(config),'--config-name','resolved_config','benchmark_catalog='+str(root/'ycb_objects/calibration_data/object_benchmark_all_targets_v1.json'),'hydra.job.chdir=false','test=True','task=Tracking','num_envs=1','train.ppo.num_actors=1','vector_mode=sync','device_id=-1','viewer=false','imshow_cam=false',f'{motion}_ball_mode=fixed',f'test_object={obj}',f'checkpoint_tracking={checkpoint}','checkpoint_catching=',f'output_name=quick_track_{motion}_{obj}']
 with (out/f'{motion}_{obj}.log').open('w') as f: p=subprocess.run(args,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
 print(motion,obj,'exit',p.returncode,flush=True)
 if p.returncode==0:
  d=json.loads((out/f'{motion}_{obj}.json').read_text());print('RESULT',motion,obj,d['successes'],20,flush=True)
 else: print((out/f'{motion}_{obj}.log').read_text()[-2500:],flush=True)
 return p.returncode
jobs=[(m,o) for m,(_,_,objects) in sets.items() for o in objects]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex: codes=list(ex.map(run,jobs))
print('DONE',codes,flush=True)
