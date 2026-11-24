"""CPU-only spawned physics workers. No EGL contexts during state-based teacher training."""
import multiprocessing as mp
import os
import traceback
import numpy as np


def worker(pipe,options,seed):
    env=None
    try:
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
        import random
        np.random.seed(seed);random.seed(seed)
        from .environment import AutonomousTeacherEnv
        env=AutonomousTeacherEnv(options)
        episode=0;obs,info=env.reset(seed=seed);pipe.send(('ok',(obs,info)))
        while True:
            command,payload=pipe.recv()
            if command=='close':break
            if command!='step':raise ValueError(command)
            obs,reward,terminated,truncated,info=env.step(payload)
            final=obs.copy()
            if terminated or truncated:
                episode+=1;obs,reset_info=env.reset(seed=seed+episode*100003)
                info['next_reset_sampling']=reset_info['reset_sampling']
                info['next_fixed_episode']=reset_info.get('fixed_episode',False)
            pipe.send(('ok',(obs,reward,terminated,truncated,info,final)))
    except (EOFError,BrokenPipeError):pass
    except BaseException:
        try:pipe.send(('error',traceback.format_exc()))
        except (EOFError,BrokenPipeError):pass
    finally:
        if env is not None:env.close()
        pipe.close()


class Pool:
    def __init__(self,count,options,seed):
        self.pipes=[];self.processes=[];self.count=count
        context=mp.get_context('spawn')
        try:
            observations=[];fixed=[]
            for i in range(count):
                parent,child=context.Pipe();p=context.Process(target=worker,args=(child,options,seed+i*1009),daemon=True)
                p.start();child.close();self.pipes.append(parent);self.processes.append(p)
                obs,info=self.receive(parent);observations.append(obs);fixed.append(info.get('fixed_episode',False))
            self.obs=np.stack(observations);self.fixed=np.asarray(fixed,dtype=bool)
        except BaseException:self.close();raise
    def receive(self,p):
        if not p.poll(180):raise TimeoutError('Autonomous physics worker timed out')
        try:status,value=p.recv()
        except EOFError:raise RuntimeError('Autonomous physics worker crashed; inspect worker log') from None
        if status!='ok':raise RuntimeError(value)
        return value
    def step(self,actions):
        for pipe,action in zip(self.pipes,actions):pipe.send(('step',action))
        result=[self.receive(p) for p in self.pipes]
        obs,reward,terminated,truncated,info,final=zip(*result)
        self.obs=np.stack(obs)
        self.fixed=np.asarray([v.get('next_fixed_episode',v.get('fixed_episode',False)) for v in info],dtype=bool)
        return self.obs,np.asarray(reward),np.asarray(terminated),np.asarray(truncated),list(info),np.stack(final)
    def close(self):
        for p in self.pipes:
            try:p.send(('close',None))
            except (EOFError,BrokenPipeError,OSError):pass
        for p in self.processes:
            p.join(timeout=2)
            if p.is_alive():p.terminate();p.join(timeout=2)
        for p in self.pipes:p.close()
