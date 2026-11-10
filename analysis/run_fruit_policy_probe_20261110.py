import subprocess,concurrent.futures,os,pathlib,json
root=pathlib.Path('/home/isee/Catch/catch_it_copy');out=root.parent/'analysis_artifacts/fruit_joint_launch_20261110'
def run(job):
 obj,case,distance=job;checkpoint=root.parent/'11-6/roll_track_best_reward_25.16.pth';folder=root/'outputs/roll_seven_object_tracking_v1_seed123/2026-11-04/01:33:28'
 env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MUJOCO_GL='egl',EVAL_MOTION='roll',EVAL_OBJECT=obj,EVAL_CHECKPOINT=str(checkpoint),EVAL_RESULT=str(out/f'policy_{obj}_{case}.json'))
 args=['/home/isee/miniconda3/envs/dcmm/bin/python','analysis/fruit_launch_policy_probe_20261110.py','--config-path',str(folder),'--config-name','resolved_config','benchmark_catalog='+str(root/'ycb_objects/calibration_data/object_benchmark_all_targets_v1.json'),'hydra.job.chdir=false','test=True','task=Tracking','num_envs=4','train.ppo.num_actors=4','seed=41000','+benchmark_options.domain_randomization=physical_dr_v1','+benchmark_options.distance_m='+distance,'vector_mode=sync','device_id=-1','viewer=false','imshow_cam=false','roll_ball_mode=fixed','test_object='+obj,'checkpoint_tracking='+str(checkpoint),'checkpoint_catching=','output_name=fruit_launch_probe_'+obj+'_'+case]
 with (out/f'policy_{obj}_{case}.log').open('w') as f:p=subprocess.run(args,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
 print(obj,case,p.returncode,flush=True)
 if p.returncode:print((out/f'policy_{obj}_{case}.log').read_text()[-2000:],flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,[(o,c,d) for o in ['apple','lemon'] for c,d in [('current','[2.0,2.2]'),('candidate','[1.9,2.1]')]]))
