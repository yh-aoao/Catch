"""Read-only deterministic Track probe, 20 episodes per ball, two physical draws."""
import sys, json, hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from gym_dcmm.envs.DcmmVecEnv_bounce_july22 import DcmmVecEnv as BounceEnv
from gym_dcmm.algs.ppo_dcmm_july22.ppo_dcmm_track import PPO_Track

original_step=BounceEnv.step
def diagnostic_step(self,action):
    obs,r,t,u,info=original_step(self,action)
    info['probe_palm_touch']=bool(self.step_touch)
    return obs,r,t,u,info
BounceEnv.step=diagnostic_step

@torch.inference_mode()
def evaluate(self):
    self.set_eval();torch.set_num_threads(1)
    n=self.num_actors;quota=10
    seeds=[int(np.random.SeedSequence([41000,i,0]).generate_state(1)[0]) for i in range(n)]
    obs,_=self.env.reset(seed=seeds)
    counts=np.zeros(n,dtype=int);rows=[]
    draws=self.env.call('benchmark_randomization')
    names=self.env.call('benchmark_object')
    for step in range(500*quota):
        actions=self.model_act({'obs':self.obs2tensor(obs)},inference=True)['actions'].clamp(-1,1)
        actions=torch.nn.functional.pad(actions,(0,self.full_action_dim-actions.shape[1]))
        obs,r,t,u,infos=self.env.step(self.action2dict(actions))
        for i in np.flatnonzero(t|u):
            if counts[i]>=quota:continue
            f=infos['final_info'][i]
            rows.append(dict(object=names[i],slot=int(i),episode=int(counts[i]),seed=seeds[i],
                truncation_flag=bool(u[i]),palm_contact=bool(f['probe_palm_touch']),
                contact_without_failure=bool(f['probe_palm_touch'] and not t[i]),
                terminated=bool(t[i]),truncated=bool(u[i]),env_time=float(f['env_time']),
                reason=f.get('terminated_reason'),physics=draws[i]))
            counts[i]+=1
            if counts[i]<quota:
                seeds[i]=int(np.random.SeedSequence([41000,int(i),int(counts[i])]).generate_state(1)[0])
                one,_=self.env.envs[i].reset(seed=seeds[i])
                for k,v in one.items():
                    if isinstance(v,dict):
                        for key,value in v.items():obs[k][key][i]=value
                    else:obs[k][i]=v
        if (counts==quota).all():break
    assert (counts==quota).all(),counts
    out=Path(__file__).parent
    (out/'episodes.jsonl').write_text(''.join(json.dumps(row,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())+'\n' for row in rows))
    summary=[]
    for name in sorted(set(names)):
        selected=[r for r in rows if r['object']==name]
        summary.append(dict(object=name,episodes=len(selected),**{k:sum(r[k] for r in selected) for k in ('truncation_flag','palm_contact','contact_without_failure','terminated')}))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'COMPLETE').write_text(hashlib.sha256((ROOT.parent/'bounce_track_best_reward_33.25.pth').read_bytes()).hexdigest())
    print(json.dumps(summary,indent=2),flush=True)
PPO_Track.test=evaluate
from train_DCMM import main
main()
