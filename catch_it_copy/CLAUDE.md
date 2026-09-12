# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 当前研究决策（2026-09-12）

用户已确定走**多任务移动操作**：先训练出 bounce、roll、basket 的可用模型，
同时调研将多个强化学习策略整合成一个模型的方法。当前推进 bounce 参数泛化、
roll/basket 改善和多任务强化学习文献调研。MoE 尚未定案。
后续讨论以 [当前研究主线与决策记录](../Codex_当前研究主线与决策记录.md) 为准，
不要沿用旧文档把“专精 bounce”视为用户当前选题。

## 本仓库目标

本仓库 fork 自 ICRA 2025 *"Catch It!"* 论文的官方实现。**核心任务：在保留原始 XArm6 机械臂的基础上，新增多种物体运动模式（throw / roll / bounce / basket）来训练和测试移动灵巧手抓取能力。**

原论文仅有 throw（抛掷）一种运动模式。本仓库扩展为四种，每种模式有独立的物理参数、奖励函数和成功判定逻辑。

> **姐妹仓库**：[`catch_it/`](../catch_it/) — 将 XArm6 替换为 UR5e，专注于机械臂替换可行性验证。

## 快速命令

```bash
# 键盘手动测试（不同运动模式）
cd gym_dcmm/envs
python3 DcmmVecEnv.py --viewer                         # throw 模式（默认）
python3 DcmmVecEnv.py --viewer --object_motion roll      # roll 模式
python3 DcmmVecEnv.py --viewer --object_motion bounce    # bounce 模式
python3 DcmmVecEnv.py --viewer --object_motion basket    # throw_basket 模式

# 训练（从仓库根目录运行）
python3 train_DCMM.py test=False task=Tracking num_envs=32 object_motion=throw
python3 train_DCMM.py test=False task=Tracking num_envs=32 object_motion=bounce
python3 train_DCMM.py test=False task=Tracking num_envs=32 object_motion=throw_basket
python3 train_DCMM.py test=False task=Catching_TwoStage num_envs=32 \
  checkpoint_tracking=outputs/xxx.pth object_motion=bounce

# 测试
python3 train_DCMM.py test=True task=Tracking num_envs=1 \
  checkpoint_tracking=assets/models/track.pth viewer=True object_motion=roll
python3 train_DCMM.py test=True task=Catching_TwoStage num_envs=1 \
  checkpoint_catching=assets/models/catch_two_stage.pth viewer=True object_motion=bounce

# 无头训练不会创建渲染器（代码内已处理，spawn 多进程下避免 GL 上下文崩溃）
# train_DCMM.py 里 os.environ['MUJOCO_GL'] = 'egl' 已注释，勿手动加回
```

训练约束：`num_envs * horizon_length = n * minibatch_size`（n 为正整数）。默认 horizon_length=64, minibatch_size=512 → `num_envs=32` 对应 `32×64=2048 = 4×512`。

## 相对于上游的核心改动

### 1. 新增 `object_motion` 参数

`train_DCMM.py` 和 `configs/config.yaml` 新增 `object_motion` 字段，支持四种模式：

| 模式 | 说明 | config 值 |
|------|------|-----------|
| throw | 原始模式：物体以初速度抛出，3D 自由飞行 | `"throw"` |
| roll | 小球在台面上滚动，滚下台后接住落球 | `"roll"` |
| bounce | 小球具有弹性，落地后弹跳，手在空中抓取弹跳中的球 | `"bounce"` |
| basket | 抛球入篮：手掌握球后抛向斜放的篮筐 | `"throw_basket"` |

### 2. 配置文件 — [configs/env/DcmmCfg.py](configs/env/DcmmCfg.py)

这是本仓库改动最大的文件：

- **保留 XArm6**：`XML_DCMM_LEAP_OBJECT_PATH = "urdf/x1_xarm6_leap_right_object.xml"`，`XML_ARM_PATH = "urdf/xarm6_right.xml"`
- **arm_joints**：`[0.0, 0.0, -0.0, 3.07, 2.25, -1.5]`（XArm6 原始值）
- **新增 `roll_arm_joints`**：roll 模式的臂姿配置
- **新增 bounce 物理参数段**（`bounce_*`）：弹性系数（COR=0.8）、接触时间、阻尼、摩擦、质量、半径、初始高度/速度
- **新增 roll 物理参数段**（`roll_*`）：台面参数（桌面 2.4m×3.2m，高度 z=0.46m）、跟踪奖励权重、失败判定阈值、抓取判定条件
- **新增 bounce 奖励/判定参数段**（`bounce_*`）：3D 位置奖励、掌心朝向奖励、速度匹配奖励、阶段切换阈值、终止判定
- **PID 参数**：XArm6 使用 `Kp_arm = [300, 400, 400, 50, 200, 20]`

### 3. 环境文件 — [gym_dcmm/envs/DcmmVecEnv.py](gym_dcmm/envs/DcmmVecEnv.py)

- `object_motion` 分支：所有模式相关逻辑使用 `if self.object_motion in ("roll", "bounce")`
- roll/bounce 模式下物体初始化的不同逻辑
- 每种模式独立的奖励函数计算、阶段切换、成功/失败判定
- 手指预置姿态（bounce 模式中跟踪阶段手指预先弯曲）
- 备用/实验文件：`DcmmVecEnv_bounce.py`, `DcmmVecEnv_roll1.py`, `DcmmVecEnv_roll2.py`, `DcmmVecEnv_new.py`, `DcmmVecEnv_record.py`

### 4. PPO 算法文件改动

`ppo_dcmm_track.py`, `ppo_dcmm_catch_two_stage.py`, `ppo_dcmm_catch_one_stage.py` 均有改动（旧版备份为 `*_old.py`）。

### 5. train_DCMM.py 差异

- 新增 `object_motion=config.object_motion` 传入 `gym.make_vec()`
- 新增确定性种子支持（`seed`, `torch_deterministic`）

## 架构概览

### 配置系统（双层，并非纯 Hydra）

- **`configs/config.yaml`**：顶层 Hydra 配置 — `task`, `num_envs`, `test`, `object_motion`, checkpoint 路径, wandb 设置。通过 `defaults` 引入 `configs/train/DcmmPPO.yaml`。
- **`configs/env/DcmmCfg.py`**：直接 import 的 Python 常量 — URDF 路径、PID 增益、奖励权重、随机化范围、各运动模式的物理参数。**不由 Hydra 管理**，通过 `getattr(DcmmCfg, 'key', default)` 读取。修改后立即生效。

### 机器人控制栈

```
Policy (PPO) → action dict {base(2), arm(6), hand(12)}
  → DcmmVecEnv._step_mujoco_simulation()
    → MJ_DCMM:
      - base: IKBase steer/drive → PID → 4 steer motors + 4 drive motors
      - arm:  IK (QP solver, zxy euler deltas) → PID → 6 arm motors
      - hand: direct PID → 16 hand motors
  → 20 MuJoCo steps per policy step (steps_per_policy)
```

`MJ_DCMM`（`agents/MujocoDcmm.py`）持有两个 MuJoCo 模型：完整机器人（`model`）和独立机械臂模型（`model_arm`）。臂 IK 在臂专用模型中求解以提高效率。

### 观察/动作空间

- **Tracking**（18D obs, 8D act）：base_v(2) + ee_pos/quat/v(10) + obj_pos/v(6)。Act：base(2) + arm_delta_pose(6)。
- **Catching**（30D obs, 20D act）：Tracking obs + hand_joints(12)。Act：Tracking act + hand_delta(12)。

臂的 delta 姿态使用 zxy 欧拉角约定：`R.from_euler('zxy', delta_pose[3:6])`，默认 ±0.025 rad/步。

### 两阶段训练流程

1. **Stage 1（Tracking）**：训练 `PPO_Track` → 通过 `models_track.py`（单分支 ActorCritic）学习 base + arm 跟踪
2. **Stage 2（Catching）**：加载 tracking checkpoint 到 `PPO_Catch_TwoStage` → 冻结 tracking 分支（`actor_mlp_t`, `mu_t`, `sigma_t`），训练 catching 分支（`actor_mlp_c`）+ critic。使用 `models_catch.py`（双分支 ActorCritic）

## 各运动模式详解

### Throw 模式（原始）

与上游一致。物体以初速度抛出，3D 轨迹飞行。奖励基于位置误差 + 姿态对齐。这是最成熟的模式。

### Bounce 模式（弹跳小球）

**物理设定**：小球从 0.9m 高度释放，具有弹性，落地后多次弹跳。物理参数分两类（`DcmmCfg.bounce_*`）：

| 类型 | 参数 | 值 |
|------|------|-----|
| 固定 | 弹性系数 COR | 0.8 |
| 固定 | 接触时间常数 | 0.008 s |
| 固定 | 地面摩擦 | [0.50, 0.05, 0.015] |
| 固定 | 初始高度 / 水平速度 / 竖直速度 | 0.9 m / 1.0 m/s / -0.2 m/s |
| 随机化 | 小球质量 | [0.04, 0.06] kg |
| 随机化 | 小球半径 | [0.038, 0.042] m |
| 随机化 | 空气阻尼 | [0.00015, 0.00025] |
| 随机化 | 手部滑动摩擦 | [1.8, 2.2]（扭转/滚动固定 0.5/0.1） |

> **Domain Randomization 是 bounce 的学术卖点**：原论文 throw 泛化"形状"（几何泛化），bounce 泛化"物理参数"（动力学泛化）。训练用范围随机采样，测试固定参数画泛化曲线。范围已收窄，避免球弹太高/变成地面滚。

**奖励设计**（tracking 阶段）：
- 3D 位置奖励（高斯 `bounce_w_3d`）+ 3D 靠近增量（`bounce_w_approach`）
- 掌心朝向球体 + 手指方向 + 速度匹配 + 协同改善
- 手指协同奖励（sync / 方向惩罚 / 关节链）

**成功/失败判定**（重要）：
- **碰手指也算成功**：tracking 阶段球常先碰手指，`step_touch` 接受手指接触（`mask_hand`）
- 只有球碰地面/障碍（`mask_coll`）→ `early_contact` 失败
- 越界（|x|>1.2 / 球到车后 / 落地 z<0.5·radius）→ `out_of_bounds` 失败
- 抓取成功要求：掌心接触 + 球速连续降低 + 手指闭合（MCP>0.3）+ 球离地

**踩坑记录**：

| 问题 | 原因 | 解决 |
|------|------|------|
| 手指抓取混乱，但成功率高 | 手指初始随机化 + 成功判定只看球接触手掌 | 去掉手指随机化，修改成功判定要求手指闭合 |
| 小球碰到手掌弹开，手指不抓 | 缺乏鼓励闭合的奖励信号 + 从全开到闭合需要 60+ 步，球 1-2 步就弹走 | 添加手指闭合奖励函数，跟踪阶段手指预先弯曲 |
| 成功率 100% 但实际失败 | PPO 循环用 truncates（超时）统计，不是 info['success'] | success 统计改用 infos['success'] |
| 参数泛化后球弹太高/变地面滚 | 随机范围太大 | 收窄范围 + 固定弹性/高度/速度 |

**当前状态**：track 成功率尚可，catch 成功率 ~99%（viewer 目测抓取成功）。

**手部摩擦**：bounce reset 时自动设置灵巧手 16 个碰撞几何体摩擦为较高值（滑动摩擦随机 [1.8, 2.2]，扭转/滚动固定），让小球碰手掌时消耗动能、减缓弹飞。参数在 `DcmmCfg.bounce_hand_friction`。

### Roll 模式（台面滚动小球 → 滚下接住）

**物理设定**：小球在台面（`roll_table_height=0.46`m，中心 [0, 2.5, 0.44]，台面 2.4m×3.2m）上朝小车滚动，滚到桌边（前缘 y≈0.9）后落下，机械臂在桌下接住滚落的小球。**底座可动**（`roll_fix_base=False`，初始 `roll_base_init_y=-0.8`），车需要前移到落点下方。

**策略演变**：

| 阶段 | 策略 | 结论 |
|------|------|------|
| 1~4 | 掌心朝下/朝上舀球、底盘转动侧面够球 | ❌ 运动学/手伸不到地面 |
| 5 | 桌面滚动，手在台面上方拦截 | 🔄 训练中 |
| 6 | **球滚下接住**（当前）：车在桌下等球滚落，接住落球 | 🔄 训练中 |

**当前策略（方案 B'：预测落点 + 追球）**：
- 球在桌面上（z > 0.46）时：按球速预测球滚到桌边（y=0.9）的落点 x，目标 = `(x_landing, roll_landing_y=0.7)`，手提前到落点拦截
- 球掉下后：直接追球本身

**奖励设计**（8.7 简化后，对齐 throw）：
- 核心只有 `reward_approach`（靠近增量，`roll_w_approach=5.0`）
- 加 `reward_touch` + `reward_ctrl` + 碰撞/限位惩罚
- 注：`reward_xy`/`reward_height`/`reward_table_h`/`reward_palm_face`/`reward_finger_dir` 代码里仍计算，但最终 `reward_pos_component = reward_approach` 全被丢弃（dead code）

**失败判定**：球越界（|x|>1.2）、滚太远（y>4.1）、或落地（z<0.05）→ `out_of_bounds`。球从前缘 y<0.8 滚下不判失败（就是要接住它）。

**当前状态**：roll 是最难的模式，track 仍在训练（此前小车总远离小球/撞桌，成功率 0）。

### Throw_Basket 模式（抛球入篮）🆕

**物理设定**：
- 篮筐中心 [0.0, 2.2, 0.9]（arm_base 前方约 2.2m、高 0.9m），半径 `basket_radius=0.20`m，绕 X 轴倾斜 `basket_tilt_deg=25°`（开口朝小车）
- 小球半径 0.04m、质量 0.05kg，橙色醒目
- 持球阶段 0.3 秒，球粘在手掌；之后以手掌速度 + 抛掷初速度释放
- **底座可动**（`basket_fix_base=False`），底盘参与瞄准篮筐

**手部姿态**：初始松握（grip，MCP=0.6），释放时手指张开。手部不抓取，主要通过臂 + 底盘瞄准和抛球。

**奖励设计**（tracking 阶段）：
- 入篮距离奖励（高斯，`basket_w_dist=10.0`）+ 靠近增量（`basket_w_approach=5.0`）+ 入篮成功（`basket_w_score=100.0`）
- 篮筐上方奖励（`basket_w_above=2.0`，鼓励从上方接近）
- 抛球引导三件套：`reward_release`（离手 +5）+ `reward_release_dir`（朝篮筐方向抛）+ `reward_forward`（抛得够远）
- **新增四项（抛球更准/更自然）**：
  - `reward_apex`：球近篮筐水平时高度略高于筐 0.2m（抛物线正确，从上方落入）
  - `reward_speed_ok`：出手速度约 3 m/s（能飞到筐，不过大过小）
  - `reward_smooth`：base/arm 动作差分惩罚（抛球动作平滑自然）
  - `reward_base_aim`：底座前进方向对准篮筐（底盘配合瞄准）
- 控制惩罚（轻量：base 0.1 / arm 0.5 / hand 0.1）

**终止判定**：
- 成功：球入篮（穿过篮筐平面且平面内距离 < 0.20m）
- 失败：球落地（z < 0）、球飞太远（距篮筐 > 2.5m）
- 超时：4 秒

**使用方式**：
```bash
# 训练（直接用 Catching 或 Tracking 任务）
python3 train_DCMM.py test=False task=Tracking num_envs=32 object_motion=throw_basket
# 或简写
python3 train_DCMM.py test=False task=Tracking num_envs=32 object_motion=basket

# 测试
python3 train_DCMM.py test=True task=Tracking num_envs=1 \
  checkpoint_tracking=outputs/xxx.pth viewer=True object_motion=throw_basket
```

**配置参数**：见 [DcmmCfg.py](configs/env/DcmmCfg.py) 中 `basket_*` 开头的所有参数，可调整篮筐位置、大小、奖励权重等。

## 调试记录与踩坑经验

参见 [进度.txt](进度.txt) 了解完整时间线。

### 通用踩坑

1. **关节限位问题**：arm j4（XML 中的 joint5）范围 [0.0, 2.65]。将初始姿态设为 j4=0.0 会导致 MuJoCo 在噪声推挤下不稳定。**始终在暖启动姿态中保持 ≥0.05 的裕度**。

2. **奖励函数设计的教训**：
   - 掌心朝下奖励用**平方非线性**（`0.25*(dot+1)²`）+ **改善增量**（`dot_now - dot_prev`），因为默认姿态下掌心 cos≈-0.63，线性奖励梯度平坦，模型无法学习
   - 手指方向奖励同样需要平方非线性和协同改善加成，否则模型会走捷径（转到 j5≈-0.13 达成掌心朝下但手指指向后方）
   - 正确的策略需要 j5≈-2.0（绕 EE 局部 X 轴旋转以倾斜掌心同时保持手指方向）

3. **成功判定必须严格**：只看"球接触手掌 + 低速 + 近距离"会导致手指张开的"假成功"。必须加入手指闭合阈值（MCP 平均 > 0.3 rad）。

4. **手指初始姿态至关重要**：在跟踪阶段手指被强制保持 0 位（全开），进入抓取阶段后才开始闭合。但对于 bounce，球接触后 1-2 步就弹走，从全开到闭合来不及。**解决方案：跟踪阶段手指预先弯曲到半闭合状态**。

5. **spawn 多进程 + MuJoCo GL 渲染冲突**：roll/basket 训练在服务器上报 `result is None`（worker 的 `env.step` 抛异常被 `except` 捕获后 `pipe.send((None, False))`，主进程 unpack None 崩溃）。根因是 async vector env 用 spawn 多进程时，每个 worker 都初始化 MuJoCo GL 渲染上下文导致崩溃。**解决：无头训练模式（`imshow_cam=False` 且无 viewer）不创建渲染器 + `render()` 开头直接返回**（代码内已处理）。勿手动加 `os.environ['MUJOCO_GL']='egl'`——那会引入 egl GLContext 无法 pickle 的新问题。

6. **checkpoint 维度不匹配（6→8）**：旧 throw tracking checkpoint 的 arm 是 4 维训练的，当前代码 arm 6 维，加载时报 `sigma_t: (6,) -> (8,)`、`mu_t.weight: (6, 128) -> (8, 128)` 等 resize 警告。根因是 checkpoint 与当前 arm 自由度不一致，需有 GPU 环境按当前 arm 维度重新训练 track 才能用于 catch。

## 无自动化测试

本项目没有自动化测试套件，验证通过在 MuJoCo viewer 中运行训练好的策略进行目视检查。

## 后续计划

### Bounce 模式
- [ ] 继续训练长时间 catch 模型，观察手指是否学会抓取弹跳球
- [ ] 如果仍然抓不住，考虑在手上增加接球辅助装置（物理挡板/网兜），或进一步降低弹性
- [ ] 优化手指闭合的奖励权重，让抓取信号更强

### Roll 模式
- [ ] 完成"球滚下接住"方案的 track 训练，让车学会预测落点并前移到位
- [ ] track 可行后继续训练 catch
- [ ] 若仍追不上，考虑降低球速 / 缩短球起始距离，或预设桌下等待点
- [ ] 长远可搜索 "catching rolling objects with dexterous hand" 相关文献找灵感

### 通用
- [ ] 统一 throw/bounce/roll 的成功判定标准，确保都有手指闭合检查
- [ ] 考虑录制评估视频的工具脚本（已有 `DcmmVecEnv_record.py`）
- [ ] 如果 bounce 和 roll 最终可行，可尝试将多模式经验迁移到 UR5e 分支（`catch_it/`）
