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

hold 是修改控制后的静态对照。cartesian 默认等待 0.5 秒、后摆 0.5 秒、前挥 0.45 秒，前挥进度 70% 时开始开手，用 0.12 秒平滑打开。以机械臂运动学模型坐标指定 y/z；根据实际模型确认方向。与策略相同，末端每步增量限制为 0.025，手部增量限制 0.15。

```bash
python3 basket_throw_probe.py --viewer --mode cartesian --swing-time 0.35 --release-fraction 0.8 --forward-y 0.16 --up-z 0.12 --log-file outputs/probe_late.jsonl
```

每次只改一个参数，先调释放时机，再调挥臂时长。过快时目标可能无法跟上，不能把目标轨迹速度当成实际球速。joint 模式可用 --joint-back 六个数、--joint-forward 六个数设置相对初始关节角（弧度）；先小幅确认方向，勿盲目放大。若球在后摆时掉落，先调整后摆和手形，不能把它当作有效投掷。

日志每策略步写 JSONL：实际/目标臂关节、球位置、实际球速、IK 成功计数、接触与释放、弹道质量、终止原因。joint 模式不经过 IK，其 ik_successes 不可解释为求解性能。文件默认覆盖，可用不同 --log-file 保留各组。

## 训练与模型测试

先查看诊断是否可产生有效球速，再从头训练修正后控制器；旧模型行为可能因控制变更改变。

```bash
python3 train_DCMM.py --config-name=basket_fixed test=False num_envs=32 viewer=false imshow_cam=false basket_log=false
python3 train_DCMM.py --config-name=basket_fixed test=True num_envs=1 checkpoint_catching=/absolute/path/model.pth viewer=true imshow_cam=false basket_log=true
```

不要追加 task=Tracking。固定底座，机械臂与手联合训练；训练不执行脚本轨迹。probe 成功不等于 PPO 已学会，若能物理抛出但学习仍停滞，下一步才实现低维轨迹学习或行为克隆预训练。

本机此前 MuJoCo DLL 无法加载，当前验证包含隔离控制测试、奖励回归、语法及 CLI 帮助；未完成真实仿真投掷或 PPO 收敛验证。
