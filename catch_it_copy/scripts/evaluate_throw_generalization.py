"""Evaluate the five-shape Throw policy; CPU inference, no training."""
import argparse,csv,hashlib,json,os,random,subprocess,sys
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--checkpoint',type=Path,required=True)
 p.add_argument('--episodes',type=int,default=200,help='Completed episodes PER SHAPE')
 p.add_argument('--instances',type=int,default=4,help='Independent creation-time physical draws per shape; run sequentially')
 p.add_argument('--shapes',nargs='+',choices=['sphere','box','cylinder','ellipsoid','capsule'],default=['sphere','box','cylinder','ellipsoid','capsule'])
 p.add_argument('--seed',type=int,default=41000)
 p.add_argument('--viewer',action='store_true',help='Use with one shape and --instances 1 on a desktop')
 p.add_argument('--output',type=Path)
 a=p.parse_args()
 if a.episodes<1 or a.instances<1 or a.episodes%a.instances:p.error('episodes must be positive and divisible by instances')
 if a.seed<0:p.error('seed must be nonnegative')
 if a.viewer and (len(a.shapes)!=1 or a.instances!=1):p.error('viewer requires one shape and --instances 1')
 checkpoint=a.checkpoint.resolve()
 if not checkpoint.is_file():p.error('Missing checkpoint: '+str(checkpoint))
 os.chdir(ROOT)
 os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
 if not a.viewer:os.environ.setdefault('MUJOCO_GL','egl')
 import numpy as np,torch
 from omegaconf import OmegaConf
 from configs.env import DcmmCfg
 from gym_dcmm.envs.DcmmVecEnv import DcmmVecEnv
 from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_catch_two_stage import PPO_Catch_TwoStage
 from ycb_objects.evaluate_trained_balls import PolicyEnvironment,batched,wilson
 torch.set_num_threads(1)
 def seed(s):np.random.seed(s);random.seed(s);torch.manual_seed(s)
 def plain(x):
  if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
  if isinstance(x,(list,tuple)):return [plain(v) for v in x]
  if isinstance(x,np.ndarray):return x.tolist()
  if isinstance(x,np.generic):return x.item()
  return x
 output=(a.output or ROOT/'evaluation_results'/datetime.now().strftime('throw_generalization_%Y%m%d_%H%M%S_%f')).resolve()
 output.mkdir(parents=True,exist_ok=False)
 cmd=[sys.executable,'train_DCMM.py','task=Catching_TwoStage','object_motion=throw','throw_generalization=multishape_v1','test=True','num_envs=1','train.ppo.num_actors=1','device_id=-1','viewer=false','imshow_cam=false','hydra.job.chdir=false','--cfg','job','--resolve']
 resolved=subprocess.run(cmd,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
 cfg=OmegaConf.create(resolved.stdout);cfg.rl_device='cpu';cfg.checkpoint_catching='';cfg.checkpoint_tracking=''
 (output/'config.yaml').write_text(OmegaConf.to_yaml(cfg))
 raw=torch.load(str(checkpoint),map_location='cpu')
 assert raw['model']['mu_t.weight'].shape[0]==8 and raw['model']['mu_c.weight'].shape[0]==12,'Expected current 8+12 action policy'
 digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
 (output/'request.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),sha256=digest,episodes_per_shape=a.episodes,instances=a.instances,shapes=a.shapes,seed=a.seed,physics_sampling='environment_creation',criterion='existing Throw info.success; not a strict stable grasp metric'),indent=2))
 saved_shapes=DcmmCfg.object_shape;results=[]
 try:
  with torch.inference_mode():
   for shape in a.shapes:
    folder=output/shape;folder.mkdir();rows=[]
    for slot in range(a.instances):
     creation_seed=int(np.random.SeedSequence([a.seed,slot,0x7A]).generate_state(1)[0]);seed(creation_seed)
     DcmmCfg.object_shape=[shape]
     env=DcmmVecEnv(task='Catching',object_motion='throw',throw_generalization='multishape_v1',object_eval=False,render_mode=None,viewer=a.viewer,imshow_cam=False,env_time=2.5,steps_per_policy=20)
     agent=None
     try:
      assert env.throw_randomization['shape']==shape
      agent=PPO_Catch_TwoStage(PolicyEnvironment(env),str(folder/('agent_'+str(slot))),cfg)
      agent.restore_test(str(checkpoint));agent.model.load_state_dict(raw['model'],strict=True)
      for key in ('running_mean_std_track','running_mean_std_hand'):getattr(agent,key).load_state_dict(raw[key],strict=True)
      agent.set_eval()
      physics=plain(env.throw_randomization)
      with (folder/('episodes_'+str(slot)+'.jsonl')).open('w') as stream:
       for episode in range(a.episodes//a.instances):
        episode_seed=int(np.random.SeedSequence([a.seed,slot,episode,0xDCC]).generate_state(1)[0]);seed(episode_seed)
        obs,_=env.reset(seed=episode_seed);total=0.
        for step in range(200):
         actions=agent.model_act({'obs':agent.obs2tensor(batched(obs))},inference=True)['actions'].clamp(-1,1)
         actions=torch.nn.functional.pad(actions,(0,agent.full_action_dim-actions.shape[1]))
         command=agent.action2dict(actions)
         obs,reward,terminated,truncated,info=env.step({k:v[0] for k,v in command.items()});total+=float(reward)
         assert np.isfinite(total) and np.isfinite(env.Dcmm.data.qpos).all()
         if terminated or truncated:break
        else:raise RuntimeError('Episode exceeded 200 steps')
        assert plain(env.throw_randomization)==physics
        row=dict(shape=shape,slot=slot,episode=episode,seed=episode_seed,success=bool(info['success']),terminated=bool(terminated),truncated=bool(truncated),reason=str(info.get('terminated_reason') or env.terminated_reason or 'timeout'),reward=total,steps=step+1,physics=physics)
        stream.write(json.dumps(row)+'\n');stream.flush();rows.append(row)
        if len(rows)%10==0:print(shape,len(rows),'/',a.episodes,flush=True)
     finally:
      if agent is not None:agent.writer.close()
      env.close()
    n=len(rows);k=sum(r['success'] for r in rows)
    result=dict(shape=shape,episodes=n,successes=k,success_rate=k/n,wilson_95_interval=wilson(k,n));results.append(result)
    (folder/'summary.json').write_text(json.dumps(result,indent=2));print(shape,f'{k}/{n} = {k/n:.1%}',flush=True)
    (output/'summary.json').write_text(json.dumps(results,indent=2))
  with (output/'summary.csv').open('w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=['shape','episodes','successes','success_rate'],extrasaction='ignore');writer.writeheader();writer.writerows(results)
  assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
  (output/'COMPLETE').write_text('All shapes completed\n');print('Results:',output,flush=True)
 finally:DcmmCfg.object_shape=saved_shapes

if __name__=='__main__':main()
