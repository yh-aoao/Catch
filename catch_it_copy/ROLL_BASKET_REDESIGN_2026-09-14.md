# Roll / Basket 奖励与控制修订

日期：2026-09-14。对应本地 `catch_it_copy` 主环境，备用环境未修改。
目标是恢复可学习的单任务策略，服务于后续多任务移动操作整合。

## 后续修复：桌面误计入整手空间

用户日志暴露 `hand_clearance=-2.020m`、`workspace=-20.501/步`。原先按
`[hand_start_id, object_id)` 选择几何体，而 XML 的桌面也位于此区间。
桌面包围球半径约 2.0001m，到自身包围盒的有符号距离为 -0.02m，
错误地产生 -2.0201m 余量。现按掌部所属 body 和手指后代 body 选择碰撞几何体，
排除桌面、物体及同一腕部 body 上的非掌部几何体；Roll 手部接触集合也使用此选择。
这是计算错误修复，不是降低空间惩罚权重。此前在此错误奖励下训练的 Roll 模型建议重训。
`base_collision` 是独立的真实接触标签，其碰撞对象仍需新版 `[roll-base-contact]` 日志确认。

## Roll Tracking

上一版将等待高度从 0.30m 降至 0.24m，并根据整手包围球继续下调；到位后永久锁定
目标。这可能使目标过低，并在轨迹变化后仍等待旧落点。

本轮恢复最大等待世界高度 0.30m，默认关闭自动下调，每步根据球当前轨迹更新
桌沿下方的预测拦截点。桌上阶段不会直接以桌面球心为追踪目标。

奖励保留：XY 对准、高度对准、有正负号的接近增量、到位速度惩罚、
桌上阶段过高惩罚、掌心朝上、整手空间不足惩罚，以及控制、碰撞、有效接触项。
空间估计仍覆盖参与碰撞的手掌和手指。真实手桌接触仍以 `table_collision` 失败。

取消“整手估计余量必须达到 3cm”才承认接触和切换阶段的硬门槛。
3cm 是软惩罚目标；球完整离开桌前沿且脱离桌面接触的条件仍保留。
这避免保守包围球误差阻断已经发生的正常接球。

## Roll Catching

两个阶段共享上述拦截塑形。保留读取真实 16 维关节的手指同步、关节链协调、
姿态误差、屈曲软限位及有上限的闭合奖励。

闭合塑形可以根据球与手的距离提前逐渐启用，不再被桌面接触或估计空间余量直接清零。
进入 grasping 后恢复较弱的真实球距离项：

`r_precision = 3 * exp(-50 * d_ee_ball²)`

此前移除此项会削弱最后贴近球的信号；采用权重 3，低于旧通用精度奖励 10。
仍有拦截与碰桌约束，不能保证每个训练种子都改善。成功阈值、回合时限未放宽。

## Basket Tracking

停车目标与奖励不变：目标为 `[basket_x, initial_arm_base_y + 0.2]`，只训练底座两维动作。

`r = -d + 20*(d_previous-d) - ||v-v_desired||² - 0.1*||action||² - 0.05`

成功额外 +100，提前失败额外 -100。目标速度为误差乘 1.5 并限速 0.8m/s，
进入目标区后参考速度为零。XY 各误差 <0.08m、速度 <0.10m/s，连续 5 个策略步成功。
停车距离、圆框随机范围、4 秒时限均保留，可继续使用兼容的 23 维观测/2 维动作模型。

## Basket Catching（抛球）

改为独立的 `parking → preparing → flight` 流程：

1. 停车期间保持臂手准备姿态，球保持在手中。连续停稳约 0.2 秒后才进入准备阶段。
2. 准备期间底座指令为零，由策略控制臂手。离开停车位置范围则退回停车，重置准备计时。
3. 准备 0.3 秒后释放。使用同一个手部附着点，避免旧版释放时球位置突跳。
   初速度取该点的有限差分世界速度加 `[0, 1.5, 2]m/s`，合速度封顶 8m/s。

分阶段奖励：

| 阶段 | 奖励 |
|---|---|
| parking | `-d + 20*(d_previous-d) - ||v-v_desired||² - 0.05` |
| preparing | `-2*predicted_error - 0.5*||arm_action||² - 0.1*||hand_action||² - 0.05` |
| flight | `5*(ball_distance_previous-ball_distance) - 0.05` |
| 事件 | 真正穿框 +100；失败或超时 -20，成功不重复扣失败罚分 |

`predicted_error` 是按当前预计释放速度，在未来 0.04–1.5 秒采样 40 个弹道点，
取离框中心的最小世界距离。它只提供稠密准备信号，不作为成功判据。
飞行阶段取消无关手姿的持续正奖励，不能靠停留或开合手指刷取抛球奖励。

每个物理子步检测球心轨迹线段是否从正面穿过倾斜圆框平面；插值求交点，
交点距框中心必须不超过 `basket_radius - basket_ball_radius`。成功奖励与统计
共用 `basket_score` 事件。穿过平面但偏离有效圆孔为 `basket_miss`。
落地、飞得过远、底盘碰撞、超时分别报告原因。持球不再受接来球的 `y<1` 条件限制。

`Catching_TwoStage` 默认冻结已加载的底座策略及 Tracking 观测归一化。
底座使用确定性均值动作，PPO 的概率和熵仅包含可训练的 18 维臂手动作。
可设置 `basket_freeze_tracking=false` 恢复联合更新；OneStage 不使用此冻结开关。
同时修复 TwoStage 尚无完整回合就记录 `best_reward=0` 的问题。

**边界：仍为辅助抛球基线。** 球在释放前被脚本附着，释放时刻固定且有速度补偿，
并未实现由手指真实释放接触决定出手时刻的自主投掷。圆框是视觉目标，成功用几何
穿框事件判断。后续若研究真实抓放，应单独实现并验证接触释放机制，不能将这里的结果当作
已经学会真实手指脱手。定时阶段暂未增加观测维度，策略需通过现有运动状态学习。

## 训练、检查和同步

在 Linux 的 `catch_it_copy` 内执行，替换占位 checkpoint 路径：

```bash
python3 train_DCMM.py test=False task=Tracking num_envs=32 object_motion=roll output_name=RollTrackV3
python3 train_DCMM.py test=False task=Catching_TwoStage num_envs=32 object_motion=roll checkpoint_tracking=/path/to/new_roll_track.pth output_name=RollCatchV3
python3 train_DCMM.py test=False task=Catching_TwoStage num_envs=32 object_motion=throw_basket checkpoint_tracking=/path/to/basket_track.pth output_name=BasketCatchV3
```

可视化测试：

```bash
python3 train_DCMM.py test=True task=Catching_TwoStage num_envs=1 object_motion=roll checkpoint_catching=/path/to/new_roll_catch.pth object_eval=false viewer=true roll_log=true
python3 train_DCMM.py test=True task=Catching_TwoStage num_envs=1 object_motion=throw_basket checkpoint_catching=/path/to/new_basket_catch.pth object_eval=false viewer=true basket_log=true
```

先确认 Roll 等待目标在桌沿下方、能接触离桌球、碰桌次数下降；Basket 日志应出现
`parking → preparing → flight`，不能一直停留在 parking。对照固定随机种子、相同回合数，
同时记录成功率、结束原因、碰撞率与奖励分项，不只比较累计 reward。

修改的奖励不会自动更新旧 checkpoint 的策略。Roll 建议重训 Track 再训 Catch；
Basket 可复用当前合适的停车模型，重训 Catch。旧回报与本轮新回报不能直接排名。

本地 CPU 回归测试覆盖实际奖励/阶段代码、坐标一致性、穿框判定、释放连续性、
冻结策略概率与归一化，以及现有 bounce 和停车回归。MuJoCo 在本机存在 DLL 初始化失败，
未完成完整仿真训练或视频效果验证，当前不声称成功率已提高。

Linux 需同步全部代码改动，尤其新增 `gym_dcmm/utils/basket_catching.py`、
`roll_rewards.py`、主环境、Python/YAML 配置、`models_catch.py` 和
`ppo_dcmm_catch_two_stage.py`。仅复制环境文件不够。
