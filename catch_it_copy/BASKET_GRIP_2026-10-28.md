# Basket 持球姿势实测（2026-10-28）

结论：球在等待过程中向手指滑动确实存在。当前较好的方案是保留机械臂初始姿态、拇指姿态和球的生成参考点，将三根手指适度弯曲为更深的杯状支撑。不是越靠掌心、握得越紧越好；过紧的方案虽然稳定，但释放和射程反而变差。

## 修改

- 新配置 `configs/basket_hold_grip.yaml`：三个手指的屈曲目标从 `[0.6,0.5,0.5]` 改为 `[0.8,0.7,0.7]`（弧度）。仍为真实接触持球，无球体绑定、外力助抛。
- `gym_dcmm/utils/basket_grip.py`：统一初始姿态和持球控制目标；支持独立配置机械臂偏移、手指、拇指和局部生成参考点，拒绝越限关节目标。
- 环境记录最后接触时的球体局部位置，用于定位滑动和释放差异。
- `basket_pose_eval_override=true`：仅用于 hold 阶段测试，明确打印旧权重训练姿态和本次测试姿态的差异，只允许四项姿态配置差异；其他任务配置仍严格校验。训练时不允许启用。
- 原 `basket_hold` 配置保持原手型，旧固定基座模型的默认手型也保持原值。

## 实验

先筛选 111 种机械臂姿态、手型和生成参考点组合，再测试若干候选的等待后抛球。最终对原手型 p00 和新手型 p04 做配对复测。使用相同种子、相同控制器、相同物理与奖励配置。

实际使用的已训练权重：
`outputs/BasketHold_03_06/2026-10-28/11:38:10/nn/throw_basket_catch_best_reward_-3.42.pth`

这里没有进行长时间重新训练，结果衡量改变持球姿态本身带来的效果。

| 指标 | 原手型 | 新手型 |
|---|---:|---:|
| 0.3～2 秒等待区间内球体局部位置变化范围，10 种子平均 | 23.85 mm | 5.60 mm |
| 等待 0.4 秒后抛球，安全进框 | 0/10 | 5/10 |
| 等待 1 秒后抛球，安全进框 | 0/10 | 5/10 |
| 等待 2 秒后抛球，安全进框 | 0/10 | 3/10 |
| 三档等待合计安全进框 | 0/30 | 13/30 |
| 三档等待合计平均朝目标方向的自由飞行距离 | 0.331 m | 0.599 m |
| 危险机器人碰撞回合 | 0/30 | 1/30（机械臂自碰撞） |

位置变化范围定义：各局部坐标在等待区间内的最大值减最小值，再取三维向量范数；不是轨迹累计路程。局部坐标使用 link6 旋转矩阵与原点，Y 正方向朝向手指，不是掌心几何中心。原手型典型种子从初始位置向手指移动约 7 cm；新手型最终停留在更靠近掌心前缘的位置。

同一策略收到不同状态会产生不同动作，因此还做了独立物理对照：固定同一组机械臂动作和开手时序，等待 1 秒，5 个种子。原手型平均自由飞行距离 0.292 m，新手型 0.660 m；两组均无危险机器人碰撞。这说明改进不只是策略反馈变化。

另测新手型下三个其他生成参考点，各 9 回合，进框分别 2/9、3/9、2/9，平均飞行距离约 0.535、0.605、0.403 m。没有充分证据表明移动生成点优于保留原参考点，因此当前配置只修改手指。位置和手型筛选是有限搜索，不能认定找到了全局最优。

自由飞行距离来自现有发射检测器的 `best_distance`：离手后朝向目标的最大投影位移，不计地面滚动。安全进框使用 `safe_basket_success`，不是仅离手成功。新手型仍有漏框和自碰撞，尚不能认定持球抛投任务已经学好，更不能直接推断移动后的成功率。

## 先直接对比测试

在 `catch_it_copy` 目录、`dcmm` 环境运行：

```bash
python3 train_DCMM.py --config-name basket_hold_grip \
  test=True num_envs=1 vector_mode=sync device_id=0 \
  viewer=true basket_viewer_speed=1.0 basket_log=true \
  basket_pose_eval_override=true \
  checkpoint_catching=/home/isee/Catch/catch_it_copy/outputs/BasketHold_03_06/2026-10-28/11:38:10/nn/throw_basket_catch_best_reward_-3.42.pth \
  train.ppo.test_num_episodes=30 output_name=BasketHoldGrip_Test
```

这条命令测试旧权重配合新手型；默认等待仍随机采样 0.3～0.6 秒，因此与上表三个固定等待档位的统计不完全相同。要对比原手型，改成 `--config-name basket_hold` 并去掉 `basket_pose_eval_override=true`。

建议先确认新手型的视频和安全进框率，再训练。后续训练可从本次 held 模型重新初始化策略，输出到新目录，不覆盖旧权重：

```bash
python3 train_DCMM.py --config-name basket_hold_grip \
  test=False num_envs=32 vector_mode=async device_id=0 \
  viewer=false basket_log=false basket_self_collision_debug=false \
  basket_warm_start=/home/isee/Catch/catch_it_copy/outputs/BasketHold_03_06/2026-10-28/11:38:10/nn/throw_basket_catch_best_reward_-3.42.pth \
  output_name=BasketHoldGrip
```

新配置训练得到的权重按相同配置测试时，不需要 `basket_pose_eval_override`。

## 证据与验证

原始数据位于 `../analysis_artifacts/basket_grip_20261028/`：

- `screen.json`、`extended_screen.json`：111 个候选持球轨迹。
- `throw_screen.json`：84 次候选抛球对照。
- `validation.json`：20 次持球、60 次相同策略抛球、10 次相同动作对照。
- `position_validation.json`：27 次不同生成参考点对照。
- `comparison.png`：局部位置漂移与等待后射程对比。
- `reproduce_validation.py`：复现配对实验，在项目目录设置 `PYTHONPATH=.` 后运行此脚本。脚本依赖同目录的筛选结果与动作记录，会覆盖同目录 validation.json。

正式训练入口的无界面测试已跑通 3 回合：安全进框 1/3、危险碰撞 0/3；另完成双环境 CPU 的短训练检查，策略加载、采样、更新与保存均跑通。这些仅为入口检查，不作为训练效果统计依据。相关单元测试 58 项通过。
