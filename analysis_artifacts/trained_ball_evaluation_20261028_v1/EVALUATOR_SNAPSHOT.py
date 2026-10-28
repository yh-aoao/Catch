"""Exact PPO test inference on fixed balls, with terminal and contact diagnostics.

Uses the saved training configuration, actual agent observation/action adapters,
and deterministic act_inference. Does not modify rewards, models or training.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import time

import mujoco
import numpy as np
from omegaconf import OmegaConf
import torch

from ycb_objects.generalization_probe import ContactRecorder,physical_parameters

ROOT=Path(__file__).resolve().parents[1]


class PolicyEnvironment:
    """Only the attribute-query interface needed by the original PPO agent."""
    def __init__(self,env):
        self.env=env
    def call(self,name):
        return [getattr(self.env,name)]


def batched(obs):
    return {key:batched(value) if isinstance(value,dict) else np.asarray(value)[None]
            for key,value in obs.items()}


def wilson(successes,total):
    if total==0:
        return [None,None]
    z=1.959963984540054
    p=successes/total
    centre=(p+z*z/(2*total))/(1+z*z/total)
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total**2))/(1+z*z/total)
    return [centre-half,centre+half]


def summarize(rows):
    n=len(rows)
    k=sum(r['task_success'] for r in rows)
    return dict(episodes=n,successes=k,success_rate=k/n if n else None,
        wilson_95_interval=wilson(k,n),
        environment_successes=sum(r['environment_success'] for r in rows),
        hand_contact_episodes=sum(r['hand_contact'] for r in rows),
        continuous_hand_contact_0p3s=sum(r['max_contiguous_hand_contact_s']>=.3-1e-6 for r in rows),
        successes_with_continuous_contact_0p3s=sum(r['task_success'] and r['max_contiguous_hand_contact_s']>=.3-1e-6 for r in rows),
        reasons=dict(Counter(r['reason'] for r in rows)),
        mean_reward=float(np.mean([r['reward'] for r in rows])) if rows else None,
        mean_episode_sim_seconds=float(np.mean([r['elapsed'] for r in rows])) if rows else None)


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=['roll_track','bounce_catch'],required=True)
    parser.add_argument('--object',choices=['original','tennis_ball','baseball'],required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--episodes',type=int,default=100)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.episodes<1:
        parser.error('episodes must be positive')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg=OmegaConf.load(args.config)
    cfg.test=True
    cfg.num_envs=1
    cfg.train.ppo.num_actors=1
    cfg.rl_device='cpu'
    cfg.device_id=-1
    cfg.viewer=False
    cfg.imshow_cam=False
    cfg.checkpoint_tracking=''
    cfg.checkpoint_catching=''
    name='default' if args.object=='original' else 'ycb:'+args.object
    kwargs=dict(viewer=False,render_mode=None,object_eval=False,env_time=2.5,steps_per_policy=20)
    if args.case=='roll_track':
        assert cfg.task=='Tracking' and cfg.object_motion=='roll' and cfg.roll_physics=='physical_v1'
        assert cfg.roll_reward_version=='legacy'
        from gym_dcmm.envs.roll_physical import make_roll_physical
        from gym_dcmm.algs.ppo_dcmm_645edc4.ppo_dcmm_track import PPO_Track as Agent
        cfg.roll_material_file=str(ROOT/'ycb_objects/calibration_data/balls.roll_table_priors_v1.json')
        speed=OmegaConf.to_container(cfg.roll_speed) if OmegaConf.is_config(cfg.roll_speed) else cfg.roll_speed
        env=make_roll_physical(task='Tracking',object_motion='roll',test_object=name,
            roll_reward_version='legacy',roll_speed=speed,roll_material_file=cfg.roll_material_file,**kwargs)
        criterion='roll_eval_success: hand interception after full front-edge departure, no table/floor support; not grasping'
    else:
        assert cfg.task=='Catching_TwoStage' and cfg.object_motion=='bounce' and cfg.bounce_physics=='physical_v2'
        from gym_dcmm.envs.bounce_physical_v2 import make_bounce_physical_v2
        from gym_dcmm.algs.ppo_dcmm_5fe75d5f.ppo_dcmm_catch_two_stage import PPO_Catch_TwoStage as Agent
        cfg.bounce_material_file=str(ROOT/'ycb_objects/calibration_data/balls.three_ball_training_v1.json')
        env=make_bounce_physical_v2(task='Catching',object_motion='bounce',test_object=name,
            bounce_launch=cfg.bounce_launch,bounce_v2=OmegaConf.to_container(cfg.bounce_v2),
            bounce_material_file=cfg.bounce_material_file,**kwargs)
        criterion='bounce_catch_proximity_v2: horizon reached, no failure, grasp stage, ee distance <0.25m; not strict hold'
    agent=None
    try:
        agent=Agent(PolicyEnvironment(env),str(args.output/'agent'),cfg)
        agent.restore_test(str(args.checkpoint))
        raw=torch.load(args.checkpoint,map_location='cpu')
        agent.model.load_state_dict(raw['model'],strict=True)
        if args.case=='bounce_catch':
            assert 'running_mean_std_track' in raw and 'running_mean_std_hand' in raw
        agent.set_eval()
        material=Path(cfg.roll_material_file if args.case=='roll_track' else cfg.bounce_material_file)
        source_paths=['ycb_objects/evaluate_trained_balls.py','gym_dcmm/envs/roll_physical.py',
                      'gym_dcmm/envs/bounce_physical_v2.py','gym_dcmm/envs/DcmmVecEnv.py',
                      'gym_dcmm/utils/roll_evaluation.py']
        source_paths.extend(str(p.relative_to(ROOT)) for p in (ROOT/('gym_dcmm/algs/ppo_dcmm_645edc4' if args.case=='roll_track' else 'gym_dcmm/algs/ppo_dcmm_5fe75d5f')).glob('*.py'))
        meta=dict(case=args.case,object=args.object,checkpoint=str(args.checkpoint.resolve()),
            checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            training_config_path=str(args.config.resolve()),
            training_config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
            material_path=str(material),material_sha256=hashlib.sha256(material.read_bytes()).hexdigest(),
            material=json.loads(material.read_text()),seed_start=args.seed,episodes=args.episodes,
            policy='deterministic mean action, original PPO normalization/clamp/padding/action2dict',
            criterion=criterion,contact_sampling_seconds=.002,mujoco_version=mujoco.__version__,
            torch_version=torch.__version__,evaluation_config=OmegaConf.to_container(cfg,resolve=True),
            sources={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths})
        (args.output/'provenance.json').write_text(json.dumps(meta,indent=2))
        rows=[]
        recorder=None
        original_contacts=env._get_contacts
        def record_contacts():
            result=original_contacts()
            if recorder is not None:
                recorder.update()
            return result
        env._get_contacts=record_contacts
        started=time.monotonic()
        with (args.output/'episodes.jsonl').open('x') as file:
            for seed in range(args.seed,args.seed+args.episodes):
                recorder=None
                torch.manual_seed(seed)
                obs,_=env.reset(seed=seed)
                recorder=ContactRecorder(env)
                initial=dict(position=env.object_pos3d.tolist(),velocity=env.object_vel6d.tolist())
                reward_sum=0.
                for step in range(70):
                    tensor=agent.obs2tensor(batched(obs))
                    actions=agent.model_act({'obs':tensor},inference=True)['actions'].clamp(-1,1)
                    actions=torch.nn.functional.pad(actions,(0,agent.full_action_dim-actions.shape[1]),value=0)
                    command=agent.action2dict(actions)
                    obs,reward,terminated,truncated,info=env.step({k:v[0] for k,v in command.items()})
                    reward_sum+=float(reward)
                    if terminated or truncated:
                        break
                else:
                    raise RuntimeError('Episode did not terminate within 70 policy steps')
                assert np.isfinite(env.Dcmm.data.qpos).all() and np.isfinite(reward_sum)
                row=dict(seed=seed,task_success=bool(info['roll_eval_success'] if args.case=='roll_track' else info['success']),
                    environment_success=bool(info.get('success',False)),
                    success_version=info.get('success_version','roll_interception'),
                    reason=str(info.get('terminated_reason') or env.terminated_reason or 'timeout'),
                    hand_contact=recorder.had_hand,max_contiguous_hand_contact_s=recorder.max_hand_duration,
                    floor_contact_at_end=recorder.floor_contact,floor_impacts=len(recorder.impacts),
                    stage=env.stage,ee_distance=float(info['ee_distance']),
                    reward=reward_sum,elapsed=float(info['env_time']),steps=step+1,
                    initial=initial,final_state=recorder.state(),physics=physical_parameters(env))
                file.write(json.dumps(row)+'\n')
                file.flush()
                rows.append(row)
                if len(rows)%10==0 or len(rows)==args.episodes:
                    summary=dict(summarize(rows),wall_seconds=time.monotonic()-started,case=args.case,object=args.object)
                    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
                    print('[eval-progress]',json.dumps(summary),flush=True)
        (args.output/'COMPLETE').write_text('completed\n')
    finally:
        if agent is not None:
            agent.writer.close()
        env.close()


if __name__=='__main__':
    main()
