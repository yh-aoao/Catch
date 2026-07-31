# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 本仓库目标

本仓库 fork 自 ICRA 2025 *"Catch It!"* 论文的官方实现。**核心任务：将原项目的 XArm6 机械臂替换为 UR5e，测试替换后的可行性，并在有问题时修改代码进行适配。**

原论文系统：AgileX Ranger Mini V2 底盘 + **XArm6** 机械臂 + LEAP Hand 灵巧手 → 在空中接住飞行物体。本仓库改为 **UR5e**。

> **姐妹仓库**：[`catch_it_copy/`](../catch_it_copy/) — 保留 XArm6，专注于新增 roll/bounce 等多种运动模式。

## 环境与安装

```bash
conda create -n dcmm python=3.8
conda activate dcmm
pip install torch torchvision torchaudio
cd catch_it && pip install -e .
pip install -r requirements.txt
```

关键依赖：`gymnasium==0.29.1`, `mujoco>=3.0.0`, `hydra-core`, `qpsolvers`（IK 求解）, `numpy-quaternion`, `tensorboardX`, `wandb`。

若出现 `mujoco.FatalError: gladLoadGL error`，在 `main()` 前加：
```python
os.environ['MUJOCO_GL'] = 'egl'
```
已在 `train_DCMM.py:19` 注释掉。

## XArm6 → UR5e 替换要点

以下 4 个文件是本仓库相对于上游的核心修改：

### 1. 模型路径与参数 — [configs/env/DcmmCfg.py](configs/env/DcmmCfg.py)

- **URDF 路径**：`XML_DCMM_LEAP_OBJECT_PATH` 从 `x1_xarm6_leap_right_object.xml` 改为 `x1_ur5e_leap_right_object.xml`；`XML_ARM_PATH` 从 `xarm6_right.xml` 改为 `ur5e_right.xml`（独立机械臂模型用于无重力 IK 求解）
- **初始关节角**：UR5e 的初始关节角与 XArm6 完全不同 → `[-1.5021, -1.8165, 1.8391, -2.2738, -1.4934, -0.0100]`
- **PID 参数**：Kp/Ki/Kd 以及输出限幅均针对 UR5e 重新调参（UR5e 的 PID 要求与 XArm6 差异较大）

### 2. 末端执行器 Body 名称 — 全局替换

UR5e 的末端执行器 body 名称为 `wrist_3_link`，而 XArm6 为 `link6`。以下文件中所有对末端执行器位姿/速度的引用都需要替换：

- [gym_dcmm/envs/DcmmVecEnv.py](gym_dcmm/envs/DcmmVecEnv.py) — 观测计算（`_get_relative_ee_pos`, `_get_relative_ee_quat`, `_get_relative_ee_v_lin_3d`）、奖励计算中的 `ee_distance`、info 字典
- [gym_dcmm/agents/MujocoDcmm.py](gym_dcmm/agents/MujocoDcmm.py) — `current_ee_pos` / `current_ee_quat` 的初始化和每步更新
- [gym_dcmm/utils/ik_pkg/ik_arm.py](gym_dcmm/utils/ik_pkg/ik_arm.py) — QP 和 LM_Chan 两种 IK 求解器中的雅可比计算和目标位姿

### 3. 路径修复 — [gym_dcmm/envs/DcmmVecEnv.py](gym_dcmm/envs/DcmmVecEnv.py#L1-L2)

在文件开头添加了 `sys.path.insert(0, '/home/isee/catch_it/catch_it')` 以修复直接运行时的 import 问题。

### 4. 新增的 URDF 模型 — [assets/urdf/](assets/urdf/)

- `x1_ur5e_leap_right_object.xml` — UR5e 完整场景模型（底盘+UR5e 臂+LEAP Hand+物体）
- `ur5e_right.xml` — 独立的 UR5e 机械臂模型，用于无重力 IK 求解

### 5. 训练物体过滤 — [configs/env/DcmmCfg.py](configs/env/DcmmCfg.py#L95-L97)

新增 `train_object_filter` 参数，用于限制训练时的物体形状，让灵巧手专注学习抓取特定形状的物体：

```python
# 只用小球训练
train_object_filter = ["sphere"]
# 恢复使用全部形状
train_object_filter = None
```

该过滤在 `_reset_object()`（[DcmmVecEnv.py:518-522](gym_dcmm/envs/DcmmVecEnv.py#L518-L522)）中生效，仅在 `object_train=True`（即 `object_eval=False`，默认训练模式）时起作用。

## 常用命令

### 测试训练好的模型
```bash
# 测试两阶段接住（使用本仓库训练好的 UR5e 模型）
python3 train_DCMM.py test=True task=Catching_TwoStage num_envs=1 \
  checkpoint_catching=outputs/Dcmm/2026-06-12/21:17:54/nn/best_reward_185.90.pth \
  object_eval=False viewer=True imshow_cam=False
```

### 训练
```bash
# Stage 1：训练 tracking
python3 train_DCMM.py test=False task=Tracking num_envs=32

# Stage 2：训练 catching（需 tracking checkpoint）
python3 train_DCMM.py test=False task=Catching_TwoStage num_envs=32 \
  checkpoint_tracking=assets/models/track.pth
```

### 键盘手动测试环境
```bash
cd gym_dcmm/envs && python3 DcmmVecEnv.py --viewer
```

### PPO 训练约束
`num_envs * horizon_length = n * minibatch_size` 必须成立（n 为正整数）。默认 horizon_length=64, minibatch_size=512，所以 `num_envs=32` → `32×64=2048 = 4×512`。

## 架构概览

### 入口：`train_DCMM.py`
Hydra 驱动的主脚本。通过 `gym.make_vec("gym_dcmm/DcmmVecWorld-v0")` 创建向量化环境，根据 `task` 选择三种 PPO 之一：`Tracking` → `PPO_Track`，`Catching_TwoStage` → `PPO_Catch_TwoStage`，`Catching_OneStage` → `PPO_Catch_OneStage`。

**注意**：catch_it 的 `train_DCMM.py` **没有** `object_motion` 参数，仅支持原始 throw 模式。运动模式扩展见 `catch_it_copy/`。

### `gym_dcmm/` 关键模块

- **`envs/DcmmVecEnv.py`** — 核心 Gym 环境。44 维观测（base 速度、臂 ee 位姿/速度/关节、12 维手部关节、物体位姿/速度），18 维动作（base 2d 速度、臂 delta xyz+roll、12 维手部 delta 关节）。两阶段接住：tracking 阶段（base+臂跟踪物体，手部静止），grasping 阶段（ee 距物体 < 0.25m 时切换）。包含领域随机化（物体物理参数、噪声、PID 增益、延迟）。

- **`agents/MujocoDcmm.py`** — `MJ_DCMM` 类封装 MuJoCo 模型加载、4 个 PID 控制器（驱动/转向/臂/手）、IK 求解器（臂用 QP 数值 IK，四轮独立转向底盘用解析 IK）、相机投影。

- **`algs/ppo_dcmm/`** — PPO 实现。`ppo_dcmm_track.py`（Tracking）、`ppo_dcmm_catch_two_stage.py`（两阶段，加载冻结的 tracking MLP 再训练 catching MLP）、`ppo_dcmm_catch_one_stage.py`（单阶段从头训）。`experience.py`（GAE 轨迹缓冲）、`models_track.py`/`models_catch.py`（Actor-Critic MLP）。

- **`utils/`** — `pid.py`（PID 控制器）、`ik_pkg/ik_arm.py`（QP 数值 IK）、`ik_pkg/ik_base.py`（底盘运动学）、`util.py`（四元数运算、延迟缓冲、变换）。

### 配置系统：`configs/`
三层 Hydra 配置：`config.yaml`（任务、模式、环境数、wandb）、`train/DcmmPPO.yaml`（PPO 超参数）、`env/DcmmCfg.py`（奖励权重、PID 增益、随机化范围、URDF 路径）。命令行覆盖使用 `key=value` 语法。

### 资产
- `assets/urdf/` — MuJoCo XML 模型，当前使用 `x1_ur5e_leap_right_object.xml`
- `assets/meshes/` — 各机器人连杆、物体、场景元素的 STL/OBJ 网格
- `assets/models/` — 预训练权重（`track.pth`、`catch_two_stage.pth`）
- `assets/objects/` — 未见过的评估物体 STL

## 调试记录与踩坑经验

参见 [进度.txt](进度.txt) 了解完整时间线。

### 关键时间节点

| 日期 | 事件 | 结论 |
|------|------|------|
| 6.17 | UR5e 替换后的模型训练跑完 | Track 模型可训练 |
| 7.8 | 测试 catching 时灵巧手表现异常 | 初步判断灵巧手安装模型（hand model attachment）有问题 |
| 7.11 | 灵巧手加了挡板（参考原模型） | 不加挡板挡不住球 |
| 7.17-18 | 用 32 环境训练 track，效果尚可 | Track 任务基本可行 |
| 7.20-21 | Catch 任务效果很差 | **结论：UR5e 替换后 catch 任务表现很差，可能这个机械臂模型不太适合此任务** |

### 已知问题

1. **灵巧手建模问题**：UR5e 的灵巧手安装模型（attachment）有问题，导致抓取时手指行为异常。即使 track 任务表现尚可，catch 任务中灵巧手仍然无法正常工作。

2. **Track OK, Catch 差**：UR5e 的 tracking 训练效果还算可以，但 catching 训练效果很差。可能原因：
   - UR5e 的运动学特性（关节范围、可达空间）与 XArm6 不同，导致抓取姿态受限
   - 灵巧手与 UR5e 末端连接处的建模不正确
   - PID 参数虽然针对 UR5e 调整过，但在抓取阶段可能仍不理想

3. **无自动化测试**：本项目没有自动化测试套件，验证通过在 MuJoCo viewer 中运行训练好的策略进行目视检查。

## 后续计划

1. **修复灵巧手安装模型**：最优先事项。需要检查 UR5e 末端 `wrist_3_link` 到 LEAP Hand 的 XML attachment 定义是否正确（关节映射、位姿偏移等）。
2. **如果建模修复后 catch 仍不行**：考虑 UR5e 是否从根本上不适合此任务——UR5e 的关节范围和工作空间与 XArm6 有本质差异，可能需要换回 XArm6 或尝试其他机械臂型号。
3. **参考 catch_it_copy 的经验**：如果后续要在 UR5e 上扩展运动模式（roll/bounce），可以参考 `catch_it_copy/` 中已有的奖励函数设计和阶段切换逻辑。
