"""Production Catch setup and inference, with exact per-instance episode quotas."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gym_dcmm.algs.ppo_dcmm_645edc4 import ppo_dcmm_catch_two_stage as roll
from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_track import terminal_metrics as bounce_metrics
from gym_dcmm.algs.ppo_dcmm_5fe75d5f.ppo_dcmm_catch_hold import PPO_Catch_Hold

def plain(x):
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    return x

def seed_for(base,slot,episode):
    return int(np.random.SeedSequence([base,slot,episode,0xDCC]).generate_state(1)[0])

def replace_obs(batch,one,i):
    for k,v in one.items():
        if isinstance(v,dict):replace_obs(batch[k],v,i)
        else:batch[k][i]=v

def initial_state(vector,i):
    env=vector.envs[i].unwrapped;model=env.Dcmm.model;data=env.Dcmm.data
    bid=model.body('object').id;j=int(model.body_jntadr[bid]);q=int(model.jnt_qposadr[j]);v=int(model.jnt_dofadr[j])
    return dict(qpos=data.qpos[q:q+7].tolist(),qvel=data.qvel[v:v+6].tolist())

def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [center-half,center+half]

@torch.inference_mode()
def evaluate(self):
    request=json.loads(Path(os.environ['MULTITARGET_EVAL_REQUEST']).read_text())
    folder=Path(request['case_dir']);n=request['instances'];quota=request['episodes']//n
    assert self.num_actors==n
    self.set_eval();torch.set_num_threads(1)
    draws=plain(self.env.call('benchmark_randomization')) if request['physics']=='dr' else [None]*n
    if request['physics']=='dr':
        assert len({d['size_scale'] for d in draws})==n
        for d in draws:
            for k,lo,hi in [('size_scale',.95,1.05),('mass_scale',.9,1.1),('friction_scale',.85,1.15),('rolling_scale',.8,1.2),('damping_scale',.9,1.1)]:
                assert lo<=d[k]<=hi,(k,d)
    seeds=[seed_for(request['seed'],i,0) for i in range(n)]
    obs,_=self.env.reset(seed=seeds)
    initial=[initial_state(self.env,i) for i in range(n)]
    count=np.zeros(n,dtype=int);length=np.zeros(n,dtype=int);returns=np.zeros(n)
    successes=stable=0;reasons={};episode_draws=[];started=time.monotonic()
    metrics=roll.terminal_metrics if request['motion']=='roll' else bounce_metrics
    with (folder/'episodes.jsonl').open('w') as stream:
        for _ in range(quota*500):
            actions=self.model_act({'obs':self.obs2tensor(obs)},inference=True)['actions'].clamp(-1,1)
            actions=torch.nn.functional.pad(actions,(0,self.full_action_dim-actions.shape[1]))
            obs,r,term,trunc,infos=self.env.step(self.action2dict(actions))
            length+=1;returns+=r;done=term|trunc
            ok,why=metrics(infos,done,trunc)
            for i in np.flatnonzero(done):
                if count[i]<quota:
                    final=infos['final_info'][i];physics=plain(final['object_benchmark'])
                    if request['physics']=='dr':assert physics['randomization']==draws[i]
                    elif request['physics']=='dr_episode':
                        draw=physics['randomization']
                        assert draw['version']=='physical_dr_episode_v1' and draw['sampling']=='episode_reset'
                        for k,lo,hi in [('size_scale',.95,1.05),('mass_scale',.9,1.1),('friction_scale',.85,1.15),('rolling_scale',.8,1.2),('damping_scale',.9,1.1)]:
                            assert lo<=draw[k]<=hi,(k,draw)
                        episode_draws.append(draw)
                    else:assert 'randomization' not in physics
                    held=bool(final.get('roll_stable_hold',False));successes+=int(ok[i]);stable+=int(held)
                    reason=str(why[i]);reasons[reason]=reasons.get(reason,0)+1
                    row=dict(slot=int(i),episode=int(count[i]),episode_seed=seeds[i],initial=initial[i],
                        success=bool(ok[i]),stable_hold=held,reason=reason,steps=int(length[i]),reward=float(returns[i]),
                        success_version=final.get('success_version'),physics=physics,
                        roll_evaluation=plain(final.get('roll_eval')),
                        bounce_diagnostics=plain({k:v for k,v in final.items() if k.startswith('bounce_grasp')}))
                    stream.write(json.dumps(row)+'\n');stream.flush();count[i]+=1
                    if count[i]<quota:
                        seeds[i]=seed_for(request['seed'],int(i),int(count[i]))
                        one,_=self.env.envs[i].reset(seed=seeds[i]);replace_obs(obs,one,int(i))
                        initial[i]=initial_state(self.env,int(i))
                length[i]=0;returns[i]=0.
            completed=int(count.sum())
            if completed and completed//20!=getattr(self,'_eval_last_print',-1):
                self._eval_last_print=completed//20
                print(f"[evaluation] {request['name']} {request['object']} {completed}/{request['episodes']} successes={successes}",flush=True)
            if np.all(count==quota):break
    assert np.all(count==quota),count
    checkpoint=Path(request['checkpoint'])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==request['checkpoint_sha256'],'Checkpoint changed during evaluation'
    result=dict(request,completed=True,episodes=int(count.sum()),successes=successes,
        success_rate=successes/request['episodes'],wilson_95_interval=wilson(successes,request['episodes']),
        stable_hold_episodes=stable if request['motion']=='roll' else None,
        stable_hold_rate=stable/request['episodes'] if request['motion']=='roll' else None,
        reasons=reasons,parameter_samples=episode_draws if request['physics']=='dr_episode' else draws,wall_seconds=time.monotonic()-started)
    temporary=folder/'summary.json.tmp';temporary.write_text(json.dumps(result,indent=2));os.replace(temporary,folder/'summary.json')
    print('[evaluation complete]',successes,'/',request['episodes'],flush=True)

roll.PPO_Catch_TwoStage.test=evaluate
PPO_Catch_Hold.test=evaluate
if __name__=='__main__':
    from train_DCMM import main
    main()
