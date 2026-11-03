import subprocess,concurrent.futures,os,json
from pathlib import Path
root=Path('/home/isee/Catch');out=root/'analysis_artifacts/object_benchmark_20261103'
def run(name):
 env=dict(os.environ,PATH='/home/isee/miniconda3/envs/dcmm/bin:'+os.environ['PATH'],OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR='/tmp/new_bounce_viewer_mpl_'+name)
 with (out/('viewer_'+name+'.log')).open('w') as f:
  p=subprocess.run(['bash','scripts/view_object_benchmark.sh','bounce',name,'--headless','--episodes','2','--output',str(out/('replay_final_'+name))],cwd=root/'catch_it_copy',env=env,stdout=f,stderr=subprocess.STDOUT)
 return dict(object=name,exit=p.returncode)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(run,['racquetball','golf_ball','softball']))
(out/'viewer_smoke.json').write_text(json.dumps(results,indent=2));print(results)
