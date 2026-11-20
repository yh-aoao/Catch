"""Evaluation-only softball size override; production catalogs untouched."""
from pathlib import Path
import os,sys,json,runpy
ROOT=Path(__file__).resolve().parents[2]/'catch_it_copy'
sys.path.insert(0,str(ROOT))
from gym_dcmm.envs import object_benchmark as benchmark
from ycb_objects.benchmark_randomization import scaled_geometry
original_geometry=benchmark.geometry
scale=float(os.environ['SOFTBALL_RADIUS_SCALE'])
def geometry(name,*args,**kwargs):
 shape=original_geometry(name,*args,**kwargs)
 if name=='softball' and scale!=1.:
  shape=scaled_geometry(shape,dict(size_scale=scale,mass_scale=1.))
 return shape
benchmark.geometry=geometry
runpy.run_path(str(ROOT/'scripts/_eval_multitarget_worker.py'),run_name='__main__')
