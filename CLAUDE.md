# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 当前研究决策（2026-09-12，用户已确定）

以**多任务移动操作**为主线：先获得 bounce、roll、basket 的可用模型，同时调研如何将
多个强化学习策略整合为一个模型。当前工作为 bounce 参数泛化、roll/basket 效果改善、
多任务强化学习策略整合文献调研。MoE 是候选，尚未确定最终方法。
后续选题建议请先参考 [当前研究主线与决策记录](Codex_当前研究主线与决策记录.md)。
此用户决策优先于旧研究文档中“优先专精 bounce”的建议。

## 工作空间总览

本工作空间基于 ICRA 2025 论文 [*Catch It! Learning to Catch in Flight with Mobile Dexterous Hands*](https://github.com/hang0610/Catch_It) 的官方实现，包含两个并行实验分支：

| 文件夹 | 机械臂 | 改动目标 | 运动模式 |
|--------|--------|----------|----------|
| [catch_it/](catch_it/) | **UR5e**（替换原 XArm6） | 测试更换机械臂模型的可行性 | throw（原始） |
| [catch_it_copy/](catch_it_copy/) | **XArm6**（原装） | 新增多种运动模式来训练 | throw / roll / bounce |

原论文系统：AgileX Ranger Mini V2 底盘 + XArm6 机械臂 + LEAP Hand 灵巧手 → 在空中接住飞行物体。

`old/` 文件夹为历史备份（原始 Catch_It 仓库、机械臂变体旧版等），**不在当前开发范围内，无需关注**。

## 两个子项目的关系

- **共同基础**：均 fork 自同一上游仓库，共享 PPO 训练框架、MuJoCo 仿真、Hydra 配置系统、IK 求解器等核心架构
- **独立演进**：两个子项目各自有独立的 git 仓库和 CLAUDE.md，改动互不影响
- **踩坑互助**：在 catch_it_copy 中积累的运动模式设计经验（奖励函数设计、阶段切换逻辑、成功判定标准）对 catch_it 后续扩展有参考价值，反之 catch_it 的机械臂替换经验也对 catch_it_copy 有参考意义

## 快速导航

- 想了解 **UR5e 机械臂替换**的细节 → 看 [catch_it/CLAUDE.md](catch_it/CLAUDE.md)
- 想了解 **多运动模式（roll/bounce）** 的细节 → 看 [catch_it_copy/CLAUDE.md](catch_it_copy/CLAUDE.md)
- 想了解 **当前进度和遗留问题** → 看各子项目的 `进度.txt`

## 环境安装（两个项目通用）

```bash
conda create -n dcmm python=3.8
conda activate dcmm
pip install torch torchvision torchaudio
cd catch_it        # 或 cd catch_it_copy
pip install -e .
pip install -r requirements.txt
```

关键依赖：`gymnasium==0.29.1`, `mujoco>=3.0.0`, `hydra-core`, `qpsolvers`, `numpy-quaternion`, `tensorboardX`, `wandb`。
