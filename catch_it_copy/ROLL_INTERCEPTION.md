# Roll 桌沿下方拦截与抓取奖励

更新：2026-09-12，根据用户提供的 7.57s 视频和当前代码继续修复。
基于 `DcmmVecEnv.py`，不涉及备用的 `DcmmVecEnv_roll*.py`。
视频约 4.26s 的关键帧可见手指伸入桌板下方、球贴近桌沿；不能仅凭视频确定每次失败原因。

## 已发现并处理的问题

- 原高度/姿态项虽有计算，但最终位置奖励只合并 XY、靠近增量和桌板惩罚。
  旧高度逻辑还混用了基座相对高度与世界桌面高度；新逻辑统一使用世界坐标。
- 抓取阶段把 12 维手部观测按 16 维关节索引读取，异常被吞掉，闭合奖励清零；
  roll 的关节链奖励也没有加进最终奖励。现在直接读取仿真真实的 16 维 qpos。
- 原 tracking→grasping 要求无接触；有效手部接触现在也能触发抓取，碰桌失败优先。
- roll Tracking 发生有效接触时，成功统计现在与 `step_touch` 对齐；失败优先。
- roll 抓取的离地判定改用世界球心高度，避免拿基座相对高度判断是否离地。

## 新的跟踪行为

球在桌面上时，预测滚出前沿后下降到拦截高度的位置。link6 最大等待世界高度由
0.30m 改为 0.24m，并根据当前整只手的上沿包围界适当下调，最低目标 0.14m。
若即使最低目标也无法提供余量，日志标记 `height_feasible=False`，不宣称目标可达。
桌上阶段到目标 10cm 内后锁定等待目标，进入 grasping 不再改成球当前位置。
未到位锁定时，仍根据轨迹更新目标。移除 roll 抓取阶段额外的 link6→球精度奖励，
避免它与等待目标冲突、再次把手拉回桌沿。

不再硬性把落点 Y 推到桌沿外 12cm：慢球在落下相同高度时未必能前进这么远。
现在使用所有参与碰撞的掌/指 geom 的包围球到桌板完整三维包围盒的距离，目标软余量
为 3cm；允许手在桌板下方留出足够竖直空间，而不是只检查 link6 的水平距离。

奖励包括 XY 位置、高度、带正负号的接近增量、到位等待时的速度惩罚、
桌上阶段抬得过高的惩罚、整手空间不足的惩罚以及掌心朝上的软奖励。手与桌板发生真实接触时，
回合以 `table_collision` 结束并扣分；球与桌面的正常滚动接触不受此项影响。
这些是奖励引导及碰撞终止；锁定的是奖励目标，没有固定机器人位姿或屏蔽动作。
包围球是保守近似，只反映当前手姿，不能保证未来弯曲扫掠路径无碰撞，也不是全臂避碰规划器。
预测使用简化的匀速滚动加自由落体模型，强摩擦减速等情况下会有误差。

## 新的抓取奖励

Catching 的 tracking 和 grasping 都使用同一组可调手指项：

- 三指对应屈曲关节的同步惩罚。
- 单根手指各屈曲关节的比例协调惩罚。
- 远处维持适度张开的准备姿态，靠近球或接触时逐渐鼓励包裹姿态。
- 闭合奖励有上限，超过软屈曲上限或反向弯曲扣分，避免只靠越弯越深刷分。

有效接触奖励、Tracking 接触完成、tracking→grasping 切换和闭合奖励共用条件：
**球完整离开桌子前沿且不再接触桌面，整手包围界留足 3cm 余量。**
桌边提前触球不会给有效接触奖励或允许主动闭合；原始接触另记为 `roll_raw_touch`。

这些是软约束，关节物理限位仍由原模型决定。原抓取成功的速度、持续步数、掌心接触、
XY 距离和闭合阈值仍保留；没有通过降低这些阈值来提高成功率。

## 调整与训练

参数在 `configs/env/DcmmCfg.py`，不直接接受 Hydra 命令行覆盖：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `roll_wait_height` | 0.24m | link6 最大等待世界高度 |
| `roll_min_wait_height` | 0.14m | 等待高度下限 |
| `roll_grasp_clearance` | 0.03m | 整手到桌板的软余量 |
| `roll_wait_lock_distance` | 0.10m | 到位锁定目标的距离 |
| `roll_intercept_ball_offset` | 0.04m | 球心相对 link6 的目标高度差 |
| `roll_w_h` | 2.0 | 高度奖励权重 |
| `roll_hand_close_distance` | 0.18m | 开始鼓励闭合的接近距离 |
| `roll_hand_ready_target` | [0.35, 0.20, 0.20]rad | 三指准备姿态 |
| `roll_hand_close_target` | [0.80, 0.60, 0.50]rad | 三指包裹姿态 |
| `roll_hand_flex_max` | 1.4rad | 屈曲软上限 |

新的奖励不会改变旧 checkpoint 已学到的行为，建议重新训练 Tracking，再用新的
Tracking 模型训练 Catching_TwoStage。观测和动作维度未改，但旧策略未必适合新目标。
注意：当前 TwoStage 代码会更新 Tracking 分支及其归一化统计，并非冻结后只训练手指。

```bash
python3 train_DCMM.py test=False task=Tracking num_envs=32 object_motion=roll output_name=RollBelowTableTrack
python3 train_DCMM.py test=False task=Catching_TwoStage num_envs=32 object_motion=roll checkpoint_tracking=/path/to/new_roll_track.pth output_name=RollBelowTableCatch
python3 train_DCMM.py test=True task=Catching_TwoStage num_envs=1 object_motion=roll checkpoint_catching=/path/to/roll_catch.pth viewer=true object_eval=false roll_log=true
```

`info['roll_reward_terms']` 提供实际分项，`roll_target_world` 提供世界目标，
`roll_waiting` 标识桌上等待阶段。`roll_log=true` 每 10 步及结束时打印 `[roll-check]`：
等待目标、是否锁定、球是否离桌、整手估计余量、原始/有效接触、稳定步数和终止原因。
`hand_clearance` 是保守几何估计；`table_collision` 才表示检测到了真实手桌接触。
直接环境入口对应 `--roll_log`。

本机只完成 CPU 回归测试，
未运行完整 MuJoCo 训练；这些修改不构成已验证的成功率提升。
Linux 运行时需同步环境、Python/YAML 配置、训练入口和 `gym_dcmm/utils/roll_rewards.py`，
避免新参数与代码不匹配。本轮 basket Catching 仅审查，见 `BASKET_CATCH_AUDIT.md`。
