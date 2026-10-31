"""Read-only deterministic evaluation of the specified fixed Basket carry model.

Run from catch_it_copy with PYTHONPATH set to that project directory.
Uses the original PPO normalization/actions and the saved training protocol.
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

from gym_dcmm.envs.DcmmVecEnv import DcmmVecEnv
from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_catch_one_stage import PPO_Catch_OneStage


ROOT = Path(__file__).resolve().parents[1]/'catch_it_copy'
RUN = ROOT/'outputs/BasketAimCarry/2026-10-29/01:23:05'


class PolicyEnvironment:
    def __init__(self, env):
        self.env = env

    def call(self, key):
        return [getattr(self.env, key)]


def batch(obs):
    return {k:batch(v) if isinstance(v,dict) else np.asarray(v)[None]
            for k,v in obs.items()}


def plain(value):
    if OmegaConf.is_config(value):
        return plain(OmegaConf.to_container(value,resolve=True))
    if isinstance(value, dict):
        return {k:plain(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [plain(v) for v in value]
    if isinstance(value,np.ndarray):
        return value.tolist()
    if isinstance(value,np.generic):
        return value.item()
    return value


def wilson(k,n):
    z=1.959963984540054
    p=k/n
    centre=(p+z*z/(2*n))/(1+z*z/n)
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [centre-half,centre+half]


def summary(rows):
    n=len(rows)
    k=sum(r['success'] for r in rows)
    return dict(episodes=n,successes=k,success_rate=k/n,wilson_95_interval=wilson(k,n),
        basket_scores=sum(r['reason']=='basket_score' for r in rows),
        basket_hits=sum(r['basket_hit'] for r in rows),
        safe_basket_successes=sum(r['safe_basket_success'] for r in rows),
        confirmed_launches=sum(r['launch_success'] for r in rows),
        safe_launches=sum(r['safe_launch_success'] for r in rows),
        released_episodes=sum(r['release_count']>0 for r in rows),
        hand_contact_episodes=sum(r['had_hand_contact'] for r in rows),
        dangerous_robot_collision_episodes=sum(r['robot_dangerous_collision'] for r in rows),
        reasons=dict(Counter(r['reason'] for r in rows)),
        mean_reward=float(np.mean([r['reward'] for r in rows])),
        mean_policy_sim_seconds=float(np.mean([r['elapsed'] for r in rows])))


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=RUN)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--episodes',type=int,default=100)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    checkpoint=args.checkpoint or args.run/'nn/throw_basket_catch_best_reward_38.38.pth'
    if args.episodes<=0:parser.error('episodes must be positive')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg=OmegaConf.load(args.run/'resolved_config.yaml')
    assert cfg.task=='Catching_OneStage' and cfg.object_motion=='throw_basket'
    assert cfg.basket_reward_version=='basket_aim_v1'
    assert cfg.basket_fixed_base_training and cfg.basket_fixed_physical and cfg.basket_carry_reward
    assert cfg.basket_curriculum_stage==3 and cfg.basket_arm_control=='joint_velocity'
    assert cfg.basket_control_probe=='off' and not cfg.basket_pose_eval_override
    assert cfg.basket_eval_hand_flex is None
    cfg.test=True
    cfg.num_envs=cfg.train.ppo.num_actors=1
    cfg.device_id=-1
    cfg.rl_device='cpu'
    cfg.viewer=cfg.imshow_cam=False
    cfg.checkpoint_tracking=cfg.checkpoint_catching=''
    env_kwargs=dict(task='Catching',camera_name=['top'],render_per_step=False,
        render_mode=None,object_name='object',img_size=cfg.train.ppo.img_dim,
        imshow_cam=False,viewer=False,object_eval=cfg.object_eval,
        env_time=2.5,steps_per_policy=20,object_motion=cfg.object_motion,
        bounce_physics=cfg.bounce_physics,bounce_launch=cfg.bounce_launch,
        bounce_log=False,basket_log=False,roll_log=False,
        basket_self_collision_debug=bool(cfg.basket_self_collision_debug),
        basket_fixed_base_training=cfg.basket_fixed_base_training,
        basket_control_probe=str(cfg.basket_control_probe),
        basket_arm_control=str(cfg.basket_arm_control),
        basket_eval_hand_flex=None,
        basket_fixed_hand_flex=list(cfg.basket_fixed_hand_flex) if cfg.basket_fixed_hand_flex is not None else None,
        basket_fixed_arm_drive_bonus=float(cfg.basket_fixed_arm_drive_bonus),
        basket_fixed_physical=bool(cfg.basket_fixed_physical),
        basket_carry_reward=bool(cfg.basket_carry_reward),
        basket_carry_options=OmegaConf.to_container(cfg.basket_carry_options,resolve=True),
        basket_reward_version=str(cfg.basket_reward_version),
        basket_mobile_options=None,basket_curriculum_stage=int(cfg.basket_curriculum_stage))
    env=DcmmVecEnv(**env_kwargs)
    agent=None
    try:
        agent=PPO_Catch_OneStage(PolicyEnvironment(env),str(args.output/'agent'),cfg)
        agent.restore_test(str(checkpoint))
        raw=torch.load(checkpoint,map_location='cpu')
        agent.model.load_state_dict(raw['model'],strict=True)
        agent.running_mean_std.load_state_dict(raw['running_mean_std'],strict=True)
        agent.set_eval()
        assert agent.actions_num==18 and agent.obs_shape==(35,)
        source_paths=['train_DCMM.py','gym_dcmm/envs/DcmmVecEnv.py','gym_dcmm/agents/MujocoDcmm.py',
            'configs/env/DcmmCfg.py','assets/urdf/x1_xarm6_leap_right_object.xml',
            'gym_dcmm/utils/basket_aim.py','gym_dcmm/utils/basket_simple.py',
            'gym_dcmm/utils/basket_physical.py','gym_dcmm/utils/basket_grip.py',
            'gym_dcmm/utils/basket_carry.py','gym_dcmm/utils/basket_mobile.py',
            'gym_dcmm/algs/ppo_dcmm/ppo_dcmm_catch_one_stage.py',
            'gym_dcmm/algs/ppo_dcmm/models_track.py','gym_dcmm/algs/ppo_dcmm/utils.py']
        meta=dict(checkpoint=str(checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            training_config=str((args.run/'resolved_config.yaml').resolve()),
            training_config_sha256=hashlib.sha256((args.run/'resolved_config.yaml').read_bytes()).hexdigest(),
            policy='Original PPO deterministic act_inference; saved running stats; clamp[-1,1]; pad right18->20 then original action2dict: fixed base zeros, arm6/hand12',
            criterion='Environment success=basket_score after confirmed physical free flight crosses effective tilted hoop aperture and .3s safety follow-through; not proximity',
            protocol='fixed_base+physical_settle+aim_carry_v1; original training initialization including fixed_hand_flex; no evaluation-only pose/hand/probe overrides; no ball force',
            seeds=[args.seed,args.seed+args.episodes-1],episodes=args.episodes,
            mujoco_version=mujoco.__version__,torch_version=torch.__version__,
            effective_config=OmegaConf.to_container(cfg,resolve=True),env_kwargs=plain(env_kwargs),
            basket_center=env.basket_center.tolist(),basket_radius=float(env.basket_cfg.basket_radius),
            ball_radius=float(env.basket_cfg.basket_ball_radius),basket_tilt_deg=float(env.basket_cfg.basket_tilt_deg),
            max_sim_time=float(env.basket_cfg.basket_max_time),policy_interval=.04,
            source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths},
            evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            historical_source_limit='No source provenance was saved in the supplied Basket run; this evaluates current local code under the saved matching protocol, not certified byte-identical historical source replay.')
        (args.output/'provenance.json').write_text(json.dumps(plain(meta),indent=2))
        rows=[]
        started=time.monotonic()
        with (args.output/'episodes.jsonl').open('x') as file:
            for seed in range(args.seed,args.seed+args.episodes):
                np.random.seed(seed);random.seed(seed);torch.manual_seed(seed)
                obs,_=env.reset(seed=seed)
                initial=dict(ball_position=env.Dcmm.data.qpos[37:40].copy().tolist(),
                    ball_velocity=env.Dcmm.data.qvel[36:39].copy().tolist(),
                    robot_position=env.Dcmm.data.qpos[:3].copy().tolist(),
                    preparation=plain(env.basket_preparation),spawn_check=plain(env.basket_spawn_check),
                    basket_center=env.basket_center.tolist())
                reward_sum=0.
                max_ball_force=0.
                for step in range(200):
                    actions=agent.model_act({'obs':agent.obs2tensor(batch(obs))},inference=True)['actions'].clamp(-1,1)
                    actions=torch.nn.functional.pad(actions,(0,agent.full_action_dim-actions.shape[1]),value=0)
                    command=agent.action2dict(actions)
                    obs,reward,terminated,truncated,info=env.step({k:v[0] for k,v in command.items()})
                    reward_sum+=float(reward)
                    max_ball_force=max(max_ball_force,abs(float(env.Dcmm.data.ctrl[-1])))
                    assert np.isfinite(env.Dcmm.data.qpos).all() and np.isfinite(reward_sum)
                    assert max_ball_force==0.,'Object force assistance appeared'
                    if terminated or truncated:break
                else:raise RuntimeError('Basket episode exceeded 200 policy steps')
                row=dict(seed=seed,success=bool(info['success']),
                    reason=str(info.get('terminated_reason') or env.terminated_reason or 'timeout'),
                    basket_success=bool(info.get('basket_success',False)),
                    basket_hit=bool(info.get('basket_hit',False)),
                    safe_basket_success=bool(info.get('safe_basket_success',False)),
                    launch_success=bool(info.get('launch_success',False)),
                    safe_launch_success=bool(info.get('safe_launch_success',False)),
                    robot_dangerous_collision=bool(info.get('robot_dangerous_collision',False)),
                    arm_self_collision=bool(info.get('arm_self_collision',False)),
                    robot_base_collision=bool(info.get('robot_base_collision',False)),
                    had_hand_contact=bool(env.basket_had_hand_contact),release_count=int(env.basket_release_count),
                    contact_total_seconds=float(env.basket_contact_total),
                    max_ball_force=max_ball_force,elapsed=float(info['env_time']),steps=step+1,reward=reward_sum,
                    terminated=bool(terminated),truncated=bool(truncated),
                    phase=env.basket_phase,initial=initial,
                    final_ball_position=env.Dcmm.data.qpos[37:40].tolist(),
                    final_ball_velocity=env.Dcmm.data.qvel[36:39].tolist(),
                    basket_control=plain(info.get('basket_control',{})),
                    release=plain(env.basket_release_snapshot),
                    simple_state=plain(env.basket_simple_state),
                    reward_totals=plain(info.get('basket_reward_totals',{})))
                assert row['success']==(row['reason']=='basket_score')
                file.write(json.dumps(row)+'\n');file.flush();rows.append(row)
                if len(rows)%10==0 or len(rows)==args.episodes:
                    result=dict(summary(rows),wall_seconds=time.monotonic()-started)
                    (args.output/'summary.json').write_text(json.dumps(result,indent=2))
                    print('[basket-eval-progress]',json.dumps(result),flush=True)
        (args.output/'COMPLETE').write_text('completed\n')
    finally:
        if agent is not None:agent.writer.close()
        env.close()


if __name__=='__main__':
    main()
