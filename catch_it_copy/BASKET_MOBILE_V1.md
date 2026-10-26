# 移动到位后投篮：basket_mobile_v1

此版本从零训练单个 PPO 策略。目标框每回合随机位置，回合内静止；底盘可以移动。
固定基座 `basket_aim_v1` 保留。新配置不加载已有投掷模型，训练入口拒绝非空训练 checkpoint。

## 控制和回合流程

1. 重置后球只放置一次；先静止 0.30 秒，让球通过重力和真实接触落稳。
2. 导航阶段：策略输出底盘 XY 速度；机械臂和手指控制器保持初始托球关节目标，屏蔽策略的臂手动作。
3. arm_base 到停车点的世界 X/Y 误差分别小于 0.10m，实际平移速度小于 0.10m/s，且球仍接触手，连续满足 0.20 秒后开放臂手控制。
4. 策略控制底盘、机械臂与手指，学习投掷。底盘仍由策略控制，速度和偏离停车点会受罚。
5. 最后手接触时必须已开放臂手，且仍满足停车位置/速度要求；随后无接触至少 0.05 秒、朝目标速度至少 0.3m/s，才是合格释放。
6. 保留真实穿越、最小自由飞行 0.1 秒/0.2m，以及命中后 0.3 秒安全观察。最终成功要求完成整个过程。

导航、减速、投掷动作来自策略。托球姿态保持、速度/加速度限制与阶段开放条件是控制约束；
不是自动驾驶，也不是策略在导航时自由学习手臂姿态。
球没有焊接、没有跟随手掌重定位、没有额外托举力、没有预设抛球速度。
未到位脱手会以 `premature_release` 失败。初始准备结束仍无手接触为 `ball_not_held`。
有合格站位但慢速掉球仍不能获得有效抛出奖励；落地/超时失败。

## 默认几何与接口

- 框中心 X 均匀采样 `[-0.4, 0.4]m`，Y 均匀采样 `[1.35, 1.85]m`，Z=0.55m。
- 半径 0.20m、球半径 0.04m、倾角25度，与近目标版本一致。
- 停车点 X 与框对齐；Y=框Y−参考框距。
- 参考框距来自原近目标Y=1.05m减去回合初始 arm_base Y，当前模型约0.934m。
  它不随当前底盘位置变化，因此不会出现停车目标跟着机器人移动的问题。
- 底盘合速度上限0.5m/s，速度指令加速度上限0.8m/s²；实际速度仍取决于机器人动力学。
- 回合最长8秒，以策略步检查超时，默认步长约0.04秒。
- 一个从零初始化的MLP，[256,128]，20维动作：底盘2＋关节速度臂6＋手12。
- 观测51维：旧35维＋机械臂关节角6、关节速度6、开放标志1、停稳进度1、手接触1、剩余时间1。
  这是状态策略，无相机输入。底盘保持原XY平移接口，未新增策略偏航动作。
- 移动版权重严格检查版本、动作语义和51/20维接口，拒绝自动裁剪加载固定基座权重。

## 奖励

| 项目 | 定义 |
|---|---|
| 靠近停车点 | `8 × (上步截断距离−当前截断距离) / max(初始距离,0.25)`；只在导航阶段及首次开放时结算 |
| 导航距离 | `−0.5 × 停车距离 × dt` |
| 导航速度误差 | `−0.5 × ||实际速度−参考速度||² × dt`；参考速度指向停车点，进入位置区后为零 |
| 持球停稳 | ＋3，一次；不终止回合 |
| 合格抛出 | ＋5，一次 |
| 前向自由飞行距离 | 累计最多＋5，0.5m封顶；仅释放后且再次接触前 |
| 弹道瞄准 | 最多＋10，一次；沿用静止框平面交点误差 |
| 实际命中 | ＋20，一次 |
| 危险碰撞 | −60，一次；保持原实际接触检测范围 |
| 任务失败 | −15，一次；包括超时、提前释放、碰撞、落地未入框 |
| 未有效抛出 | 原−5，一次，可与任务失败叠加 |
| 开放后底盘稳定 | `−2 × (速度²＋停车距离²) × dt`，包含飞行观察期 |
| 时间 | −0.15/秒 |
| 底盘控制/平滑 | `−0.02 × mean(归一化动作²) × dt`；动作变化另扣`0.02 × mean(变化²)` |
| 机械臂动作变化 | 原`−0.01 × mean(归一化动作变化²)`；导航屏蔽期间不扣 |

导航距离先截断至`[0,max(初始距离,0.25)]`，进度带正负号，来回移动不会重复获得总进度奖励。
正奖励理论最多约51；停车后不投会因未释放和任务失败扣分，不能凭停车获得成功。
碰撞成本增加到60，使新增导航奖励不会抵消危险碰撞罚分。
这些系数是待实际训练验证的初始设计，不是收敛保证。

## 从零训练

```bash
conda activate dcmm
cd /home/isee/Catch/catch_it_copy
python3 train_DCMM.py --config-name basket_mobile \
  test=False num_envs=32 vector_mode=async device_id=0 \
  viewer=false basket_log=false basket_self_collision_debug=false \
  output_name=BasketMobileV1_FromScratch
```

不要传旧的 `checkpoint_tracking` 或 `checkpoint_catching`。本次训练不做旧权重迁移。
若后续确实需要保留固定基座技能再学移动，应另做显式迁移：扩展输入、把旧18个动作正确映射至新输出的后18行、
处理归一化统计和价值网络，并分别验证；不能使用现有宽松的部分行拷贝加载。

范围等选项在 `configs/config.yaml` 的 `basket_mobile` 中，支持 Hydra 覆盖，例如：

```bash
python3 train_DCMM.py --config-name basket_mobile \
  'basket_mobile.target_x_range=[-0.2,0.2]' \
  'basket_mobile.target_y_range=[1.2,1.5]' \
  output_name=BasketMobileV1_Easy_FromScratch
```

这是另一次从零训练；不会暗中加载前次权重，也没有按worker独立自动升级范围。

## 测试和指标

```bash
python3 train_DCMM.py --config-name basket_mobile \
  test=True num_envs=1 vector_mode=sync device_id=0 \
  viewer=true basket_viewer_speed=0.25 basket_log=true \
  checkpoint_catching=/绝对路径/新移动模型.pth \
  output_name=BasketMobileV1_Test
```

默认200个完整回合，或达到100000策略步上限后结束；可覆盖`train.ppo.test_num_episodes`。
训练/测试结束显式关闭环境；仅增加清理路径，未证明解决历史viewer退出段错误。
短训练生成的smoke权重只用于检查加载，不是可用投篮模型。

日志 `basket_control` 含当前目标、停车点、停车距离、速度、阶段、出手时距离/速度。
TensorBoard/W&B记录`basket_mobile/parking_success`、`qualified_release`、`basket_hit`、
`safe_basket_success`、`robot_dangerous_collision`，统计使用完整回合的final_info，避免向量环境自动重置覆盖结果。
`mean_success`表示安全入框。停车率高而释放率低时应检查投掷学习；释放率高而入框率低时检查弹道误差。
除原best_reward外，累计至少50个完整回合后保存`throw_basket_catch_best_success.pth`，
按最多200回合窗口的安全入框统计改善时更新。它仍需独立评估，尤其是成功率为0时。

## 验证与边界（2026-10-26）

- 70项相关测试通过，包含新增15项移动任务逻辑测试、固定投掷奖励/成功判定回归、动作接口和权重拒绝检查。
- `basket_mobile_probe.py`在真实MuJoCo中检查15个种子：目标回合内静止、导航时屏蔽臂手动作、物理持球、底盘实际移动并停稳。
  诊断使用手写停车参考控制器，15/15到位且无记录的危险碰撞；它不是强化学习成功率。
- CPU同步和异步PPO从零短训练跑通，异步包括完整失败回合、参数更新、日志与权重保存。
- 新移动权重能加载并按指定2个回合结束测试。短训练仅验证执行链路，没有训练出成功策略。
- 历史`test_basket_base_tracking.py`有3项错误，在原HEAD环境代码也能复现，源于旧Tracking测试与后续物理投掷实现不一致。
  本次未更改旧Tracking任务来迎合这些测试。
- 尚未进行完整GPU训练或证明移动入框收敛；需要实际训练曲线、视频与200回合评估。

```bash
MUJOCO_GL=egl python3 basket_mobile_probe.py --episodes 15 --report /tmp/basket_mobile_probe.json
python3 -m unittest tests.test_basket_mobile tests.test_basket_aim tests.test_basket_simple \
  tests.test_basket_fixed_policy tests.test_basket_catching tests.test_basket_control \
  tests.test_basket_launch_v4 tests.test_basket_launch_v5
```
