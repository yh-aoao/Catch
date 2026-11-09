"""Four randomized physical instances per target, five episodes each."""
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gym_dcmm.algs.ppo_dcmm_645edc4 import ppo_dcmm_track as roll
from gym_dcmm.algs.ppo_dcmm_july22 import ppo_dcmm_track as bounce

def plain(x):
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    return x

@torch.no_grad()
def test(self):
    self.set_eval();obs,_=self.env.reset(seed=41000)
    draws=self.env.call('benchmark_randomization')
    assert len(draws)==4 and len({d['size_scale'] for d in draws})==4
    for d in draws:
        for key,lo,hi in [('size_scale',.95,1.05),('mass_scale',.9,1.1),('friction_scale',.85,1.15),('rolling_scale',.8,1.2),('damping_scale',.9,1.1)]:
            assert lo<=d[key]<=hi,(key,d)
    count=np.zeros(4,dtype=int);length=np.zeros(4,dtype=int);reward=np.zeros(4);records=[]
    metrics=roll.terminal_metrics if os.environ['EVAL_MOTION']=='roll' else bounce.terminal_metrics
    for _ in range(20000):
        actions=self.model_act({'obs':self.obs2tensor(obs)},inference=True)['actions'].clamp(-1,1)
        actions=torch.nn.functional.pad(actions,(0,self.full_action_dim-actions.shape[1]))
        obs,r,term,trunc,infos=self.env.step(self.action2dict(actions))
        length+=1;reward+=r;done=term|trunc
        successes,reasons=metrics(infos,done,trunc)
        for i in np.flatnonzero(done):
            if count[i]<5:
                final=infos['final_info'][i]
                assert final['object_benchmark']['randomization']==draws[i]
                records.append(dict(slot=int(i),episode=int(count[i]),success=bool(successes[i]),
                    reason=str(reasons[i]),steps=int(length[i]),reward=float(reward[i]),
                    legacy_success=bool(final.get('success',False)),physics=plain(final['object_benchmark'])))
                count[i]+=1
            length[i]=0;reward[i]=0
        if np.all(count==5):break
    assert len(records)==20 and np.all(count==5)
    result=dict(motion=os.environ['EVAL_MOTION'],object=os.environ['EVAL_OBJECT'],seed=41000,
        episodes=20,successes=sum(r['success'] for r in records),instances=plain(draws),
        records=records,checkpoint=os.environ['EVAL_CHECKPOINT'])
    Path(os.environ['EVAL_RESULT']).write_text(json.dumps(result,indent=2))
    print('DR_RESULT',result['successes'],'/20',flush=True)

roll.PPO_Track.test=test;bounce.PPO_Track.test=test
from train_DCMM import main
main()
