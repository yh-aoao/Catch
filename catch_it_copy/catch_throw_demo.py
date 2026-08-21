"""
catch → throw 顺序任务（方式 B：规则状态机 + 两个独立模型）

用法：
    python3 catch_throw_demo.py \
        catch_tracking=<接球track checkpoint> \
        catch_catching=<接球catch checkpoint> \
        basket_tracking=<抛球track checkpoint> \
        viewer=true

流程：
    1. 阶段1「接球」：用 throw 的 catch 模型（PPO_Catch_TwoStage）跑，直到 info['success']=True
    2. 阶段2「抛球」：切换到 throw_basket 的 track 模型（PPO_Track）跑，直到 info['success']=True
"""
from __future__ import annotations

import os
import sys
import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from termcolor import cprint
from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_catch_two_stage import PPO_Catch_TwoStage
from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_track import PPO_Track
import gymnasium as gym
import gym_dcmm
import numpy as np
import datetime
import pytz

OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg == '' else arg)


def build_env(config, task, object_motion):
    """构建向量化环境"""
    env_task = 'Tracking' if task == 'Tracking' else 'Catching'
    env = gym.make_vec(
        "gym_dcmm/DcmmVecWorld-v0", num_envs=1,
        task=env_task, camera_name=["top"],
        render_per_step=False, render_mode="rgb_array",
        object_name="object",
        img_size=config.train.ppo.img_dim,
        imshow_cam=config.imshow_cam,
        viewer=config.viewer,
        print_obs=False, print_info=False,
        print_reward=False, print_ctrl=False,
        print_contacts=False, object_eval=config.object_eval,
        env_time=2.5, steps_per_policy=20,
        object_motion=object_motion,
    )
    return env


def run_phase(agent, env, max_steps=150, phase_name=""):
    """运行一个阶段，返回 (是否成功, info)"""
    agent.set_eval()
    reset_obs, _ = env.reset()
    agent.obs = {'obs': agent.obs2tensor(reset_obs)}
    for step in range(max_steps):
        res_dict = agent.model_act(agent.obs, inference=True)
        actions = res_dict['actions']
        actions = torch.clamp(actions, -1, 1)
        actions = torch.nn.functional.pad(actions, (0, agent.full_action_dim - actions.size(1)), value=0)
        actions_dict = agent.action2dict(actions)
        obs, r, terminates, truncates, infos = env.step(actions_dict)
        agent.obs = {'obs': agent.obs2tensor(obs)}
        # 成功判定
        if infos.get('success', False):
            print(f"[{phase_name}] 成功，step={step}")
            return True, infos
        if terminates or truncates:
            print(f"[{phase_name}] 提前结束，step={step}, reason={env.call('terminated_reason')[0] if hasattr(env, 'call') else 'unknown'}")
            return False, infos
    print(f"[{phase_name}] 超时")
    return False, infos


@hydra.main(config_name='config', config_path='configs')
def main(config: DictConfig):
    torch.multiprocessing.set_start_method('spawn')
    config.rl_device = f'cuda:{config.device_id}' if config.device_id >= 0 else 'cpu'
    config.test = True

    # 输出目录（复用现有结构）
    output_dif = os.path.join('outputs', config.output_name, 'catch_throw')
    os.makedirs(output_dif, exist_ok=True)

    # ============ 阶段1：接球（throw 的 catch 模型）============
    cprint('阶段1：接球', 'green', attrs=['bold'])
    config.task = 'Catching_TwoStage'
    config.object_motion = 'throw'
    config.checkpoint_tracking = to_absolute_path(config.catch_tracking)
    config.checkpoint_catching = to_absolute_path(config.catch_catching)
    env_catch = build_env(config, 'Catching', 'throw')
    agent_catch = PPO_Catch_TwoStage(env_catch, output_dif, full_config=config)
    agent_catch.restore_test(to_absolute_path(config.catch_catching))
    success_catch, _ = run_phase(agent_catch, env_catch, phase_name="接球")
    env_catch.close()

    if not success_catch:
        print("接球失败，结束")
        return

    # ============ 阶段2：抛球（throw_basket 的 track 模型）============
    cprint('阶段2：抛球入筐', 'green', attrs=['bold'])
    config.task = 'Tracking'
    config.object_motion = 'throw_basket'
    env_basket = build_env(config, 'Tracking', 'throw_basket')
    agent_basket = PPO_Track(env_basket, output_dif, full_config=config)
    agent_basket.restore_test(to_absolute_path(config.basket_tracking))
    success_basket, _ = run_phase(agent_basket, env_basket, phase_name="抛球")
    env_basket.close()

    print(f"\n最终结果：接球={'成功' if success_catch else '失败'}，抛球={'成功' if success_basket else '失败'}")


if __name__ == '__main__':
    main()
