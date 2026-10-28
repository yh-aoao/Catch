import json,random
from pathlib import Path
import numpy as np,torch
from omegaconf import OmegaConf
from ycb_objects.evaluate_trained_balls import PolicyEnvironment,batched
from gym_dcmm.envs.bounce_physical_v2 import make_bounce_physical_v2
from gym_dcmm.algs.ppo_dcmm_5fe75d5f.ppo_dcmm_catch_two_stage import PPO_Catch_TwoStage as Agent
r=Path('/home/isee/Catch/bounce_three_ball_catching_v1_seed123/2026-10-28/11:36:32')
cfg=OmegaConf.load(r/'resolved_config.yaml');cfg.test=True;cfg.num_envs=1;cfg.train.ppo.num_actors=1;cfg.rl_device='cpu';cfg.device_id=-1;cfg.viewer=False;cfg.checkpoint_tracking='';cfg.checkpoint_catching=''
torch.set_num_threads(1)
results=[]
for ball in ['ycb:tennis_ball']:
 env=make_bounce_physical_v2(task='Catching',object_motion='bounce',test_object=ball,bounce_launch=cfg.bounce_launch,bounce_v2=OmegaConf.to_container(cfg.bounce_v2),bounce_material_file='/home/isee/Catch/catch_it_copy/ycb_objects/calibration_data/balls.three_ball_training_v1.json',viewer=False,render_mode=None,object_eval=False,env_time=2.5,steps_per_policy=20)
 agent=Agent(PolicyEnvironment(env),'/tmp/bounce_hand_audit_agent',cfg);agent.restore_test(str(r/'nn/bounce_catch_best_reward_288.46.pth'));agent.model.load_state_dict(torch.load(r/'nn/bounce_catch_best_reward_288.46.pth',map_location='cpu')['model'],strict=True);agent.set_eval()
 jointids=[int(np.flatnonzero(env.Dcmm.model.jnt_qposadr==21+i)[0]) for i in range(16)]
 ranges=env.Dcmm.model.jnt_range[jointids];names=[env.Dcmm.model.joint(i).name for i in jointids]
 for seed in [0,1]:
  obs,_=env.reset(seed=seed);samples=[]
  with torch.inference_mode():
   for i in range(70):
    a=agent.model_act({'obs':agent.obs2tensor(batched(obs))},inference=True)['actions'].clamp(-1,1)
    command=agent.action2dict(a)
    obs,reward,t,u,info=env.step({k:v[0] for k,v in command.items()})
    q=env.Dcmm.data.qpos[21:37].copy();target=env.Dcmm.target_hand_qpos.copy()
    samples.append(dict(t=float(info['env_time']),stage=env.stage,q=q.tolist(),target=target.tolist(),hand_action=command['hand'][0].tolist(),target_limit_excess=np.maximum(ranges[:,0]-target,np.maximum(target-ranges[:,1],0)).tolist()))
    if t or u:break
  arr=np.array([s['target_limit_excess'] for s in samples]);excess=arr.max(axis=0)
  result=dict(ball=ball,seed=seed,success=bool(info['success']),reason=env.terminated_reason,samples=samples,joint_names=names,joint_ranges=ranges.tolist(),max_target_limit_excess=excess.tolist(),policy_steps_with_out_of_range_target=int((arr.max(axis=1)>1e-6).sum()))
  results.append(result)
  print('AUDIT',ball,seed,'out-of-range steps',result['policy_steps_with_out_of_range_target'],'/',len(samples),'maxexcess',float(excess.max()),flush=True)
 agent.writer.close();env.close()
p=Path('/home/isee/Catch/analysis_artifacts/bounce_video_20261028_194408/tennis_hand_audit.json');p.write_text(json.dumps(results,indent=2))
