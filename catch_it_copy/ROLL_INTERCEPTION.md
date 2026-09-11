# Roll 桌沿下方拦截与抓取奖励

本次修复基于当前 `DcmmVecEnv.py`，不涉及备用的 `DcmmVecEnv_roll*.py`。

## 已发现并处理的问题

- 原高度/姿态项虽有计算，但最终位置奖励只合并 XY、靠近增量和桌板惩罚。
  旧高度逻辑还混用了基座相对高度与世界桌面高度；新逻辑统一使用世界坐标。
- 抓取阶段把 12 维手部观测按 16 维关节索引读取，异常被吞掉，闭合奖励清零；
  roll 的关节链奖励也没有加进最终奖励。现在直接读取仿真真实的 16 维 qpos。
- 原 tracking→grasping 要求无接触；有效手部接触现在也能触发抓取，碰桌失败优先。
- roll Tracking 发生有效接触时，成功统计现在与 `step_touch` 对齐；失败优先。
- roll 抓取的离地判定改用世界球心高度，避免拿基座相对高度判断是否离地。

## 新的跟踪行为

球在桌面上时，根据当前速度预测滚出前沿后下降到拦截高度的位置；手的 link6 目标
位于世界 Z=0.30m、桌沿外侧至少 0.12m。球刚离桌但仍较高时继续追踪下落拦截点，
不会立即把目标抬回球当前高度；球低于拦截高度后跟随下降中的球。
进入 grasping 后改为跟随球附近的位置。

奖励包括 XY 位置、高度、带正负号的接近增量、到位等待时的速度惩罚、
桌上阶段抬得过高的惩罚以及靠近桌沿内侧的惩罚。手与桌板发生真实接触时，
回合以 `table_collision` 结束并扣分；球与桌面的正常滚动接触不受此项影响。
这些是奖励引导及碰撞终止，没有把手强行固定到某个位置，也不是完整的全臂避碰规划器。
预测使用简化的匀速滚动加自由落体模型，强摩擦减速等情况下会有误差。

## 新的抓取奖励

Catching 的 tracking 和 grasping 都使用同一组可调手指项：

- 三指对应屈曲关节的同步惩罚。
- 单根手指各屈曲关节的比例协调惩罚。
- 远处维持适度张开的准备姿态，靠近球或接触时逐渐鼓励包裹姿态。
- 闭合奖励有上限，超过软屈曲上限或反向弯曲扣分，避免只靠越弯越深刷分。

这些是软约束，关节物理限位仍由原模型决定。原抓取成功的速度、持续步数、掌心接触、
XY 距离和闭合阈值仍保留；没有通过降低这些阈值来提高成功率。

## 调整与训练

参数在 `configs/env/DcmmCfg.py`，不直接接受 Hydra 命令行覆盖：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `roll_wait_height` | 0.30m | link6 等待世界高度 |
| `roll_intercept_ball_offset` | 0.04m | 球心相对 link6 的目标高度差 |
| `roll_table_clearance` | 0.12m | 桌沿前方水平余量 |
| `roll_w_h` | 2.0 | 高度奖励权重 |
| `roll_hand_close_distance` | 0.18m | 开始鼓励闭合的接近距离 |
| `roll_hand_ready_target` | [0.35, 0.20, 0.20]rad | 三指准备姿态 |
| `roll_hand_close_target` | [0.80, 0.60, 0.50]rad | 三指包裹姿态 |
| `roll_hand_flex_max` | 1.4rad | 屈曲软上限 |

新的奖励不会改变旧 checkpoint 已学到的行为，建议重新训练 Tracking，再用新的
Tracking 模型训练 Catching_TwoStage。观测和动作维度未改，但旧策略未必适合新目标。

```bash
python3 train_DCMM.py test=False task=Tracking num_envs=32 object_motion=roll output_name=RollBelowTableTrack
python3 train_DCMM.py test=False task=Catching_TwoStage num_envs=32 object_motion=roll checkpoint_tracking=/path/to/new_roll_track.pth output_name=RollBelowTableCatch
```

`info['roll_reward_terms']` 提供实际分项，`roll_target_world` 提供世界目标，
`roll_waiting` 标识桌上等待阶段，可用于后续日志分析。本机只完成 CPU 回归测试，
未运行完整 MuJoCo 训练；这些修改不构成已验证的成功率提升。
Linux 运行时需同步环境、配置、新增 `gym_dcmm/utils/roll_rewards.py`；本次 basket
还需同步 `gym_dcmm/utils/basket_tracking.py`，避免代码与配置不匹配。
