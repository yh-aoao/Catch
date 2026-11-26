"""Inference-only collection of physically reached stage-one states."""
import argparse,copy,hashlib,json,os,pickle
from pathlib import Path
import numpy as np
import torch
from .environment import AutonomousTeacherEnv
from .policy import Policy,SCHEMA
from .state import VERSION

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=500)
    p.add_argument('--max-episodes',type=int,default=3000);p.add_argument('--seed',type=int,default=900123)
    args=p.parse_args();torch.set_num_threads(1)
    if args.count<1 or args.max_episodes<args.count:raise ValueError('Invalid count/max-episodes')
    if args.output.exists():raise FileExistsError(args.output)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    options=copy.deepcopy(saved['config']['options'])
    if saved.get('schema')!=SCHEMA or options.get('two_stage')!='prepare' or options.get('preparation_level')!=2:
        raise ValueError('Collect from final-distance stage-one prepare checkpoint')
    options['evaluation_task']='prepare';model=Policy();model.load_state_dict(saved['model']);model.eval()
    env=AutonomousTeacherEnv(options);states=[];reasons={}
    try:
        for ep in range(args.max_episodes):
            obs,_=env.reset(seed=args.seed+ep)
            for _ in range(512):
                with torch.no_grad():action=model.act(torch.as_tensor(obs)[None])[0].numpy()
                obs,_,done,trunc,info=env.step(action)
                if done or trunc:break
            reason=info['terminated_reason'];reasons[reason]=reasons.get(reason,0)+1
            if info['prepare_success']:
                snapshot=env.last_ready_snapshot;snapshot['collection_seed']=args.seed+ep;states.append(snapshot)
            if (ep+1)%10==0:print(json.dumps(dict(episodes=ep+1,collected=len(states),reasons=reasons)),flush=True)
            if len(states)>=args.count:break
    finally:env.close()
    if len(states)<args.count:raise RuntimeError('Insufficient successful states; improve stage one or increase max-episodes')
    result=dict(version=VERSION,states=states,source_checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        source_options=options,episodes=ep+1,reasons=reasons,mujoco_version=__import__('mujoco').__version__)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temp=args.output.with_suffix('.tmp');temp.write_bytes(pickle.dumps(result,protocol=5));os.replace(temp,args.output)
    print(json.dumps(dict(output=str(args.output),states=len(states),episodes=ep+1),indent=2))
if __name__=='__main__':main()
