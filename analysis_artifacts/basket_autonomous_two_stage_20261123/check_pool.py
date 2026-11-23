import os,sys,json
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1';os.environ['MUJOCO_GL']='egl'
ROOT=Path('/home/isee/Catch/catch_it_copy');sys.path.insert(0,str(ROOT))
from basket_autonomous.pool import Pool
import numpy as np

def main():
    os.chdir(ROOT)
    for stage in ('fixed','mobile'):
        cfg=json.loads((ROOT/'basket_autonomous/configs'/f'{stage}_stage.json').read_text())
        opt=cfg['options'];opt['episode_seconds']=.04
        if stage=='mobile':opt['fixed_replay_probability']=0.
        pool=Pool(2,opt,123)
        try:
            assert np.all(pool.fixed==(stage=='fixed'))
            for _ in range(3):
                obs,r,term,trunc,infos,final=pool.step(np.full((2,20),.1,np.float32))
                assert obs.shape==final.shape==(2,65)
                assert np.all(pool.fixed==(stage=='fixed'))
                assert (term|trunc).all()
                assert all(i['fixed_episode']==(stage=='fixed') for i in infos)
            print('PASS',stage,'spawned workers, terminal observations, auto reset; no training',flush=True)
        finally:pool.close()
if __name__=='__main__':main()
