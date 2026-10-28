# Bounce 两阶段 Catch：controlled_hold_v1

这版从已有 `bounce_track_best_reward_19.31.pth` 初始化跟踪分支，重新训练手指分支。球的 `physical_v2` 材质、三球采样和发射条件沿用上一版。新版为显式开关，配置默认仍为 `bounce_grasp_version=legacy`。

## 训练

```bash
conda activate dcmm
cd /home/isee/Catch/catch_it_copy
bash scripts/train_bounce_three_ball_hold_v1.sh
```

默认 GPU 0、16 环境、2500 万环境步、seed 123。脚本依次查找：

- 项目 `outputs/bounce_three_ball_tracking_v1_seed123/2026-10-28/00:42:24/nn/bounce_track_best_reward_19.31.pth`
- 项目父目录下相同的训练目录，适配本地 `/home/isee/Catch/bounce_three_ball_tracking_v1_seed123/...`

服务器路径不同时可显式指定，例如：

```bash
bash scripts/train_bounce_three_ball_hold_v1.sh \
  device_id=3 \
  checkpoint_tracking=/绝对路径/bounce_track_best_reward_19.31.pth
```

输出目录为 `outputs/bounce_three_ball_catching_controlled_hold_v1_seed123/日期/时间/`，包含：

- `nn/bounce_catch_last.pth`：每次 rollout 更新后原子保存；含优化器、步数、学习率和随机状态。
- `nn/bounce_catch_best_reward_*.pth`：保留当前最佳奖励模型。
- `resolved_config.yaml`、`bounce_physics_provenance.json`、`bounce_grasp_provenance.json`：配置、物理与源码/初始化模型 SHA256。
- `bounce_grasp/episodes_*.jsonl`：分环境逐回合的奖励分项、物体物理参数、保持时间及终止原因。
- `tb/`：训练奖励与新版成功率。

第一阶段 Track 无需重新训练。前 50 万步冻结跟踪分支，之后其学习率为手指/价值分支的 0.1 倍。Track 的观测归一化固定为保存的统计值。手指动作尺度由每策略步 0.15 rad 改为 0.08 rad。

## 修改内容

### 手部控制

新类 `gym_dcmm/envs/bounce_grasp.py::BounceGraspEnv` 仅在 `physical_v2 + Catching_TwoStage + controlled_hold_v1` 中启用。继承原环境的机械臂、底盘、观测、动作延迟与碰撞/越界处理。

- 按真实关节 ID 识别 12 个可控关节。目标裁剪到关节范围内，并保留默认 0.02 rad 边界余量；窄关节按范围比例缩小余量。
- 每策略步目标增量最多 0.08 rad，目标与实际角差最多 0.35 rad，防止目标无限积累。
- 独立手部 PID 使用对测量值求导，减少目标突变的微分冲击；加积分饱和处理和总输出限幅，动作噪声后再次限幅。
- 手关节现有限位的时间常数设为 0.002 s。没有投影实际 qpos、粘球、增加球外力或修改球的材质参数。
- 默认 1 Nm 输出上限是工程仿真先验，并与 XML 限幅取交集；不是实机电机标定值。真实关节仍是软限位，需要监测实际误差。

### 奖励与几何

实际手部观测是 12 维数组，分别为三指各 3 维和拇指 3 维。新代码直接使用正确数据结构，替换旧版失效的手指奖励；不采用屈曲角越大越高的线性奖励。

几何/FK 核验表明掌面包球侧为 `link6 +Z`，手指伸出方向为 `+Y`。新奖励使用实际掌面参考偏移，腕部、相机与前臂不计入手部接触。

奖励包括：掌面区域接近、靠近增量、近球时掌面朝向/相对速度/准备开度、真实接触、同一控制时刻的手内包球/接触分布、持续受控保持，以及动作幅度/变化、关节越限、控制和碰撞惩罚。所有包球形状收益有上限；拇指不与三指强制同角度或同符号。

以上正向分数统一组成有界势能 `Phi`，实际 shaping 为 `gamma*Phi_next - Phi_previous`。gamma 从 PPO 配置传入并核对，成功、失败和超时的终态势能均归零。这样持续靠近、重复擦碰、将近抓稳后反复松开都不会通过重复正分而获利。额外发放成功奖励25、终端失败惩罚25及小步成本。日志 `sums` 是实际奖励组成；`potential_score_sums` 只是诊断分数，不会再加到总奖励。

### 成功判定

`success_version=bounce_controlled_hold_v1`。每 2 ms 控制 tick 采样，要求同时满足：

- 球在掌指捕获区域，存在有法向力的真实手部接触。
- 至少两个手部区域接触，接触分布跨度至少 0.15。
- 没有腕部、前臂、地面等外部支撑，球底距地面超过 0.02 m。
- 球相对掌部参考点速度不超过 0.35 m/s，实际关节越限不超过 0.08 rad。
- 以上条件连续保持至少 0.3 秒，且策略步末仍成立。

接触或其他条件中断后计时归零。成功可提前结束回合；碰撞或严重关节越限优先判失败。0.25 m 接近及超时不再代表成功。该判定是可测的持球标准，不是完整力封闭证明。它和旧版 proximity 成功率、奖励数值不能直接比较。

可通过配置调整新环境参数，例如 `+bounce_grasp.hold_seconds=0.5`；默认值见 `GraspConfig`。

## 恢复训练和测试

恢复生产训练的 last：

```bash
bash scripts/train_bounce_three_ball_hold_v1.sh \
  checkpoint_catching=/绝对路径/新训练目录/nn/bounce_catch_last.pth
```

恢复时需保持 `bounce_grasp_training` 参数与保存配置一致。优化器、计步、归一化和 RNG 恢复，环境从新回合开始；不是回合中途逐步精确重放。输出会新建日期/时间子目录。

训练后查看任一球：

```bash
bash scripts/test_bounce_three_ball_hold_v1.sh \
  /绝对路径/新训练目录/nn/bounce_catch_last.pth tennis_ball
```

球名可选 `original`、`tennis_ball`、`baseball`。测试读取模型保存配置、固定球类，开启 viewer；脚本固定 Hydra 工作目录，避免相对网格路径失效。

## 验证与回退

相关环境、PPO、物理与材质共 29 项测试通过。已实际执行两环境异步短训练、冻结期间真实 Track 权重/归一化逐位核对、断点恢复后解冻更新，以及新测试脚本的无窗口推理。未进行正式长训练，尚无新版训练后成功率。

用同一旧 Catch 模型、相同种子对每球各两局做控制故障对照（2 ms 采样）：

| 球类 | 旧版最大实际关节越限 | 新版最大实际关节越限 | 新版目标越限 |
|---|---:|---:|---:|
| 原始球 | 2.018 rad | 0.0144 rad | 0 |
| 网球 | 5.389 rad | 0.0144 rad | 0 |
| 棒球 | 1.425 rad | 0.0285 rad | 0 |

这是故障回归测试，不是成功率估计。验证材料：`/home/isee/Catch/analysis_artifacts/bounce_controlled_hold_validation_20261029/`。

额外运行的历史结构检查有两个既有失败；用修改前完整快照复跑得到相同失败。原共享 `DcmmVecEnv.py`、原机器人类、原 PID、原 Catch PPO 和旧版训练/测试脚本与备份哈希一致。本版未覆盖这些旧实现。

原版完整源码、配置和 Track/Catch 权重备份位于：
`backups/bounce_before_controlled_hold_20261028_195129/manifest.json`。

回退训练使用原脚本（默认 `bounce_grasp_version=legacy`）：

```bash
bash scripts/train_bounce_three_ball_v1.sh \
  task=Catching_TwoStage \
  checkpoint_tracking=/绝对路径/bounce_track_best_reward_19.31.pth \
  output_name=bounce_three_ball_catching_legacy_retry_seed123
```

旧模型 viewer 脚本仍为 `scripts/test_bounce_three_ball_catch_v1.sh`。回退不需要覆盖当前共享文件。
