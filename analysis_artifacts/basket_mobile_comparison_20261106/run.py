import os,sys,subprocess
from pathlib import Path
d=Path(__file__).resolve().parent
jobs=[]
for kind in ['joint','single']:
 for ball in ['original','tennis_ball','baseball']:
  log=open(d/(kind+'_'+ball+'.log'),'w')
  env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
  p=subprocess.Popen([sys.executable,str(d/'evaluate.py'),kind,ball],stdout=log,stderr=subprocess.STDOUT,env=env)
  jobs.append((p,log,kind,ball))
failed=[]
for p,log,kind,ball in jobs:
 code=p.wait();log.close()
 print(kind,ball,'exit',code,flush=True)
 if code:failed.append((kind,ball,code))
if failed:raise RuntimeError(failed)
