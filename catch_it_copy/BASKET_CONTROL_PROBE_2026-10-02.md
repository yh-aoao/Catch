# Basket 控制修正与挥臂测试（2026-10-02）

本次先完成执行层修正和物理挥臂诊断；低维策略学习、示范预训练尚未实现。保留现有固定底座单阶段 PPO 和弹道奖励。

## 修正

Basket 调用 move_ee_pose(measured_state=True)：实际关节状态作为 IK 种子，MuJoCo wxyz 与 SciPy xyzw 显式转换，IK 失败或非有限解时恢复内部实际状态，环境关节目标退回实际位置。其他任务使用原默认路径。日志新增 control_version=basket_measured_ik_v1、arm_target_error。

basket_throw_probe.py 不加载策略，不写球位置/速度、不绑定球。末端轨迹模式使用修正后的 IK，关节轨迹模式直接提供平滑关节目标并按模型关节范围裁剪。它们仍走原物理控制器、接触与终止逻辑。默认关节方向和幅度只是诊断起点，未经真仿真验证；不能保证默认会朝目标挥臂或成功入篮。

## 测试（在 catch_it_copy 运行）

```bash
python3 basket_throw_probe.py --viewer --mode hold --episodes 1 --log-file outputs/probe_hold.jsonl
python3 basket_throw_probe.py --viewer --mode cartesian --episodes 3 --log-file outputs/probe_cartesian.jsonl
python3 basket_throw_probe.py --viewer --mode joint --episodes 3 --log-file outputs/probe_joint.jsonl
```

hold 是修改控制后的静态对照。**2026-10-10 更新**：cartesian 默认等待 0.5 秒，取消后摆，直接前上挥动 0.45 秒，在前挥时间的 50% 处开始开手，用 0.12 秒平滑打开。以机械臂运动学模型坐标指定 y/z；根据实际模型确认方向。与策略相同，末端每步增量限制为 0.025，手部增量限制 0.15。时间比例不是实际末端位移比例，控制器也可能滞后。

```bash
python3 basket_throw_probe.py --viewer --mode cartesian --swing-time 0.35 --release-fraction 0.8 --forward-y 0.16 --up-z 0.12 --log-file outputs/probe_late.jsonl
```

每次只改一个参数，先调释放时机，再调挥臂时长。过快时目标可能无法跟上，不能把目标轨迹速度当成实际球速。joint 模式可用 --joint-back 六个数、--joint-forward 六个数设置相对初始关节角（弧度）；先小幅确认方向，勿盲目放大。若球在后摆时掉落，先调整后摆和手形，不能把它当作有效投掷。

如需复测旧轨迹，显式加 `--back-time 0.5 --release-fraction 0.7`；只有 `--back-time` 大于零才使用 back 偏移。默认关节前挥方向仍是未经物理验证的候选，不能保证它对应向上抛球。

## 本轮诊断和下一步

用户提供的日志中，Cartesian 球在约 0.73 秒最后接触手，早于原定 1.0 秒开始前挥；Joint 球在约 1.52 秒脱手时，水平速度约 0.49 m/s，竖直速度约 -1.25 m/s。这些现象先说明持球/挥臂/释放失败，不能用来认定奖励权重一定有问题。

先不带 viewer 扫描 9 组时序，避免已有的 GLX viewer 报错干扰测试：

```bash
python3 basket_throw_probe.py --mode cartesian --scan --episodes 1 --log-file outputs/probe_scan_direct.jsonl
```

扫描挥臂时长 `[0.3, 0.45, 0.6]` 秒与松手时间比例 `[0.3, 0.5, 0.7]` 的全部组合。`--episodes` 是每个组合的重复次数，设为 3 会运行 27 轮。每轮都会重新 reset；目前不是严格配对随机种子的对照实验，优选组合应多次复测。

针对一个组合观看、复测：

```bash
python3 basket_throw_probe.py --viewer --mode cartesian --episodes 3 --back-time 0 --swing-time 0.45 --release-fraction 0.5 --log-file outputs/probe_direct_view.jsonl
```

这里的 0.45/0.5 只是默认起点，请用扫描结果选值。若所有组合都在开手前掉球，先解决持球/手掌姿态；若实际末端有明显前上速度，而脱手球没有，检查球与手的接触、开手延迟；若末端实际也几乎不动，先查控制与目标跟随。

关节方向静态校准：

```bash
python3 basket_throw_probe.py --calibrate-joints --log-file outputs/probe_joint_directions.jsonl
```

仅在独立机械臂运动学模型上计算各关节 ±0.1 rad 对 link6 位置与 Y 轴方向的影响，写出 12 个候选后退出，不执行挥臂、不移动球。`ee_delta_arm_frame` 是机械臂模型坐标中的末端位移。它不能验证动态速度、完整机器人的碰撞或球是否留在手上；用它确认符号后仍需 `--mode joint --joint-forward ...` 实测。当前没有自动选方向或低维 RL。

新增 JSONL 记录类型：`config`、`trial`、`step`、`summary`，校准使用 `joint_directions`。每步包括：

- `ee_position_world` / `ee_velocity_world`：实际 link6 位置、相邻策略采样的平均速度，非目标速度或物理子步峰值。
- `link6_y_world` / `palm_up_cos`：沿用项目 link6 Y 轴作为掌心方向的约定；并非独立标定的手掌表面法线。
- `planned_open_fraction`：脚本开手进度；`hand_target` / `hand_q` 用来比较目标与实际手指响应。
- `ball_velocity`：实际物理球速，`release.last_contact` 保留物理子步的脱手前球速。

每轮 `[probe-summary]` 的 `release.status` 区分前挥前失球、计划开手前失球和开手后释放；后者也不自动表示抛成功。`upward_forward_flight` 仅检查最后接触时朝篮筐的水平速度与向上速度均大于 0.1 m/s，用来区分向前上方飞行和掉落，**不代替环境入篮成功标准**。若一轮有多次释放，summary 描述最后一次记录；完整过程查看 step.release 的 release_count。最大球速、最大末端速度分别取各自策略采样最大值，不能把两个不同时间的峰值拼成一次投掷。

`throw_valid=True` 仍只是环境的释放距离门槛；`predicted_crossing=True` 是穿过篮筐无限平面，均不代表入框。当前修改限定在诊断脚本，不改 PPO 奖励、训练动作、球物理参数或其他运动模式；不会让现有训练命令自动执行脚本挥臂。

日志每策略步写 JSONL：实际/目标臂关节、球位置、实际球速、IK 成功计数、接触与释放、弹道质量、终止原因。joint 模式不经过 IK，其 ik_successes 不可解释为求解性能。文件默认覆盖，可用不同 --log-file 保留各组。

## 训练与模型测试

先查看诊断是否可产生有效球速，再从头训练修正后控制器；旧模型行为可能因控制变更改变。

```bash
python3 train_DCMM.py --config-name=basket_fixed test=False num_envs=32 viewer=false imshow_cam=false basket_log=false
python3 train_DCMM.py --config-name=basket_fixed test=True num_envs=1 checkpoint_catching=/absolute/path/model.pth viewer=true imshow_cam=false basket_log=true
```

不要追加 task=Tracking。固定底座，机械臂与手联合训练；训练不执行脚本轨迹。probe 成功不等于 PPO 已学会，若能物理抛出但学习仍停滞，下一步才实现低维轨迹学习或行为克隆预训练。

本机此前 MuJoCo DLL 无法加载，当前验证包含隔离控制测试、奖励回归、语法及 CLI 帮助；未完成真实仿真投掷或 PPO 收敛验证。
