from pathlib import Path
import json,os,subprocess
OUT=Path(__file__).resolve().parent;ROOT=Path('/home/isee/Catch/catch_it_copy')
req=json.loads((OUT/'bounce_original_range1.0/request.json').read_text());req.update(name='bounce_original_spin0',episodes=60,case_dir=str(OUT/'bounce_original_spin0'))
folder=Path(req['case_dir']);folder.mkdir();(folder/'request.json').write_text(json.dumps(req,indent=2))
cfg=ROOT/'evaluation_results/bounce_484_last_200_new/configs/bounce_dr_best_4_84.yaml'
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl',MULTITARGET_EVAL_REQUEST=str(folder/'request.json'),PROBE_DR_SCALE='1')
cmd=['/home/isee/miniconda3/envs/dcmm/bin/python',str(OUT/'probe_worker.py'),'--config-path',str(cfg.parent),'--config-name',cfg.stem,'test_object=original','num_envs=10','train.ppo.num_actors=10','hydra.job.chdir=false','bounce_v2.spin_factor=[0,0]','output_name=diagnosis_spin0']
with (folder/'worker.log').open('w') as log:p=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
raise SystemExit(p.returncode)
