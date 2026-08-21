"""
catch → throw 顺序任务演示脚本（方式 B：规则状态机）

流程：
1. 阶段1「接球」：加载 throw 的 catch 模型，运行直到球被抓稳（info['success']=True）
2. 阶段2「抛球」：切换到 throw_basket 的 track 模型，运行直到球入筐

注意：
- 两个模型独立训练好，本脚本只是把它们串起来
- 切换时 basket 环境会 reset（球重新放到掌心），这是"近似衔接"
- 真正的无缝衔接需要传递 arm/hand 状态，后续可优化
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hydra
from omegaconf import DictConfig
import torch
import gymnasium as gym
import gym_dcmm


def run_catch(env, agent, max_steps=200):
    """阶段1：接球，返回是否成功"""
    obs, info = env.reset()
    for _ in range(max_steps):
        action = agent.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if info.get('success', False):
            return True
        if terminated or truncated:
            return False
    return False


def run_throw(env, agent, max_steps=200):
    """阶段2：抛球入筐，返回是否成功"""
    obs, info = env.reset()
    for _ in range(max_steps):
        action = agent.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if info.get('success', False):
            return True
        if terminated or truncated:
            return False
    return False


def main(catch_ckpt, basket_ckpt, viewer=False):
    # 阶段1：接球环境
    env_catch = gym.make_vec(
        "gym_dcmm/DcmmVecWorld-v0", num_envs=1,
        task="Catching", object_motion="throw",
        viewer=viewer, render_mode="rgb_array",
        camera_name=["top"], steps_per_policy=20,
    )
    # 阶段2：抛球环境
    env_basket = gym.make_vec(
        "gym_dcmm/DcmmVecWorld-v0", num_envs=1,
        task="Tracking", object_motion="throw_basket",
        viewer=viewer, render_mode="rgb_array",
        camera_name=["top"], steps_per_policy=20,
    )

    # 加载两个模型（需要对应的 PPO agent 类）
    from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_catch_two_stage import PPO_Catch_TwoStage
    from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_track import PPO_Track

    # 这里简化：只演示流程，实际需要完整的 PPO 配置
    print("[catch_throw] 阶段1：接球")
    # TODO: 加载 catch 模型并运行
    print("[catch_throw] 阶段2：抛球")
    # TODO: 加载 basket 模型并运行


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--catch_ckpt", type=str, required=True, help="接球模型 checkpoint")
    parser.add_argument("--basket_ckpt", type=str, required=True, help="抛球模型 checkpoint")
    parser.add_argument("--viewer", action="store_true", help="打开 viewer")
    args = parser.parse_args()
    main(args.catch_ckpt, args.basket_ckpt, args.viewer)
