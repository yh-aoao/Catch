# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 本仓库目标

本仓库 fork 自 ICRA 2025 *"Catch It!"* 论文的官方实现。**核心任务：在保留原始 XArm6 机械臂的基础上，新增多种物体运动模式（throw / roll / bounce）来训练和测试移动灵巧手抓取能力。**

原论文仅有 throw（抛掷）一种运动模式。本仓库扩展为三种，每种模式有独立的物理参数、奖励函数和成功判定逻辑。

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

# MuJoCo 渲染修复（若 gladLoadGL error）
# 在 train_DCMM.py 的 main() 前加 os.environ['MUJOCO_GL'] = 'egl'
```

训练约束：`num_envs * horizon_length = n * minibatch_size`（n 为正整数）。默认 horizon_length=64, minibatch_size=512 → `num_envs=32` 对应 `32×64=2048 = 4×512`。

## 相对于上游的核心改动

### 1. 新增 `object_motion` 参数

`train_DCMM.py` 和 `configs/config.yaml` 新增 `object_motion` 字段，支持三种模式：

| 模式 | 说明 | config 值 |
|------|------|-----------|
| throw | 原始模式：物体以初速度抛出，3D 自由飞行 | `"throw"` |
| roll | 小球在台面上滚动，手需要贴近台面拦截小球 | `"roll"` |
| bounce | 小球具有弹性，落地后弹跳，手在空中抓取弹跳中的球 | `"bounce"` |

### 2. 配置文件 — [configs/env/DcmmCfg.py](configs/env/DcmmCfg.py)

这是本仓库改动最大的文件：

- **保留 XArm6**：`XML_DCMM_LEAP_OBJECT_PATH = "urdf/x1_xarm6_leap_right_object.xml"`，`XML_ARM_PATH = "urdf/xarm6_right.xml"`
- **arm_joints**：`[0.0, 0.0, -0.0, 3.07, 2.25, -1.5]`（XArm6 原始值）
- **新增 `roll_arm_joints`**：roll 模式的臂姿配置
- **新增 bounce 物理参数段**（`bounce_*`）：弹性系数（COR≈0.78）、接触时间、阻尼、摩擦、质量、半径、初始高度/速度
- **新增 roll 物理参数段**（`roll_*`）：台面参数（桌面 2.4m×3.2m，高度 z=0.39m）、跟踪奖励权重、失败判定阈值、抓取判定条件
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

**物理设定**：小球从约 0.9m 高度释放，具有弹性和初始水平/竖直速度，落地后多次弹跳。弹性系数 COR≈0.78，弹跳 5~6 次。

**奖励设计**（区别于 throw）：
- 3D 位置奖励 + 靠近增量奖励
- 掌心朝向球体奖励（8.0 权重 + 3.0 额外加成当夹角 < 20°）
- 手指方向奖励（8.0 权重 + 3.0 额外加成）
- 速度匹配奖励（鼓励手部速度与球速一致）
- 协同改善加成（手掌+手指方向同时改善时额外奖励）
- 手指预置弯曲姿态（MCP 关节 0.5 rad），让手指在跟踪阶段就基本半闭合，减少抓取时从全开到闭合的延迟

**成功判定**（重要改动）：
- 球速低于 0.05 m/s 且连续 5 个控制步 → 初步判定
- **必须** MCP 关节平均屈曲超过 0.3 rad 才算真正抓取 → 防止"球停在手掌上手指张开"也被判定成功
- 进入抓取阶段后最多等 20 步，超时判定失败

**踩坑记录**：

| 问题 | 原因 | 解决 |
|------|------|------|
| 手指抓取混乱，但成功率高 | 手指初始随机化 + 成功判定只看球接触手掌 | 去掉手指随机化，修改成功判定要求手指闭合 |
| 小球碰到手掌弹开，手指不抓 | 缺乏鼓励闭合的奖励信号 + 从全开到闭合需要 60+ 步，球 1-2 步就弹走 | 添加手指闭合奖励函数，跟踪阶段手指预先弯曲 |
| Track 训练效果差（手指预弯后） | 奖励函数出问题 | 修改奖励函数，重新训练 |
| 小球弹性/速度不够，太快停在地面 | 弹性参数不合适 | 调整弹性和初始速度参数 |

**当前状态**：track 效果尚可，catch 中手指学会抓取仍有困难。

**新增手部摩擦**（7.30）：在 bounce 模式 reset 时，自动设置灵巧手所有碰撞几何体的摩擦系数为更高值（默认 `[2.0, 0.5, 0.1]`），让小球碰到手掌时消耗更多动能、减缓弹飞。参数在 `DcmmCfg.bounce_hand_friction` 中配置。

### Roll 模式（台面滚动小球）

**物理设定**：小球在台面上滚动（台面约在 z=0.39m 高度，2.4m×3.2m 范围），手需要贴近台面拦截小球。

**策略演变**：

| 阶段 | 策略 | 问题 | 结论 |
|------|------|------|------|
| 1 | 手掌朝下抓取（palm-down） | 需要翻转手腕，XArm6 运动学上很难实现 | ❌ 不可行 |
| 2 | 尝试转动灵巧手末端连接关节 | 不确定是否需要其他关节配合才能掌心朝下 | ❌ 不确定 |
| 3 | 手掌朝上放地面，小球滚上来后收紧（palm-up scooping） | 位姿无法满足，手伸不到地面 | ❌ 不可行 |
| 4 | 底盘转动，手臂从侧面够球 → 前臂更水平 → 掌心自由 | 地面上滚的球手伸不下去 | ❌ 不可行 |
| 5 | **改为桌面滚动**：在台面上滚球，手在台面上方拦截 | 当前方案 | 🔄 训练中 |

**奖励设计**（当前桌面滚动方案）：
- XY 高斯靠近奖励（`roll_w_xy=1.0`, `roll_sigma_xy=0.45`）+ XY 靠近增量（`roll_w_approach=5.0`）
- 掌心朝向球体奖励（`roll_w_palm_face=8.0` + bonus）+ 手指方向奖励 + 协同改善加成
- 跟踪阶段手指预置半闭合（MCP=0.6 rad），形成"栅栏"拦截姿态

**成功判定**：与 bounce 类似，要求手指闭合（MCP > 0.3 rad）才算真正抓取。

**当前状态**：改为桌面滚动后正在训练 track，效果待观察。Roll 是三种模式中最困难的。

**新增桌面高度奖励**（7.30）：在 roll 模式跟踪阶段新增 `reward_table_h` 奖励项，鼓励手的末端保持在桌面高度（锚点 z=0.42m），并对穿到桌面下方施加惩罚（`roll_w_below_table_penalty=-5.0`）。这有助于模型理解手应该在桌面上方拦截小球。

### Throw_Basket 模式（抛球入篮）🆕

**新增于 7.30**。这是第四种运动模式，小球初始在手掌中，机械臂通过抛掷动作将球投向篮筐。

**物理设定**：
- 篮筐中心在 arm_base 前方约 2.2m、高 1.2m 处，半径 0.15m
- 橙色可视化：扁平椭球（篮筐环）+ 圆柱（底部指示器）
- 小球半径 0.04m，质量 0.05kg，橙色醒目颜色
- 持球阶段 0.3 秒，球粘在手掌上；之后以手掌速度 + 抛掷初速度释放

**手部姿态**：手指微屈成杯状（MCP=0.4, DIP=0.3, 拇指=0.2），托住球。手部不参与抓取，主要通过臂和底盘的动作瞄准和抛球。

**奖励设计**：
- 球到篮筐中心的 3D 距离奖励（高斯型，`basket_w_dist=10.0`, `basket_sigma_dist=0.3`）
- 靠近增量奖励（`basket_w_approach=5.0`）
- 入篮成功奖励（球进入 `basket_radius` 内，`basket_w_score=50.0`）
- 篮筐上方奖励（鼓励从上方接近，`basket_w_above=2.0`）
- 控制惩罚（轻量化，底座 0.1/臂 0.5/手 0.1）

**终止判定**：
- 成功：球进入篮筐（距离 < 0.15m）
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

## 无自动化测试

本项目没有自动化测试套件，验证通过在 MuJoCo viewer 中运行训练好的策略进行目视检查。

## 后续计划

### Bounce 模式
- [ ] 继续训练长时间 catch 模型，观察手指是否学会抓取弹跳球
- [ ] 如果仍然抓不住，考虑在手上增加接球辅助装置（物理挡板/网兜），或进一步降低弹性
- [ ] 优化手指闭合的奖励权重，让抓取信号更强

### Roll 模式
- [ ] 先完成桌面滚动方案的 track 训练，评估基本可行性
- [ ] 如果桌面滚动 track 可行，继续训练 catch
- [ ] 如果仍不可行，考虑等球从桌上滚下来再在空中接住（类似 throw 但轨迹不同）
- [ ] 长远来看，可搜索 "catching rolling objects with dexterous hand" 相关文献找灵感

### 通用
- [ ] 统一 throw/bounce/roll 的成功判定标准，确保都有手指闭合检查
- [ ] 考虑录制评估视频的工具脚本（已有 `DcmmVecEnv_record.py`）
- [ ] 如果 bounce 和 roll 最终可行，可尝试将多模式经验迁移到 UR5e 分支（`catch_it/`）
