# 原进框奖励＋前送持球奖励

新增配置 `basket_aim_carry`。保留新手型、无重叠放球和重力下自然稳定准备，固定基座、35 维观测、18 维动作不变。

## 原奖励完整保留

仍调用 `basket_aim_v1` 的原奖励：有效发射最多 5 分、朝目标飞行进度最多 5 分、预测落点瞄准最多 10 分、穿框一次 20 分，以及原碰撞惩罚和机械臂动作平滑惩罚。

不再按“前送是否达到 6 cm”削减 20 分进框奖励，也不把真实安全进框记为 `basket_unqualified`。`Basket SR` 恢复为安全进框率。前送距离、肩肘速度和掌内位置另行记录。

## 新增项

全部只在真实手部接触、无危险接触、首次确认发射之前采样；离手后摆臂不能增加这些奖励。

| 项 | 默认范围 | 作用 |
|---|---:|---|
| `carry_forward` | 0～6 | 对每一段首次向前推进的距离给分，按肩肘前向速度和球前向速度平滑加权。小幅前送即可获得反馈。 |
| `carry_position` | 0～2 | 同一段有效前送中，球相对掌部的位置保持得好，额外给分。 |
| `carry_slip` | −2～0 | 首次推进时，球偏离目标位置越大，扣分越多。 |
| `carry_release` | 0～4 | 有效发射确认时只支付一次，结合最后接触时前送距离、肩肘速度与前送过程中的位置保持质量。 |

前送距离默认以 6 cm 归一化饱和，速度以 0.3 m/s 归一化，均不是硬性成功门槛。与旧版位移×速度×球速相乘的进度不同，新的距离进度单独累计，使预训练策略最初约 1 cm 的小运动也能获得奖励信号。

保持位置的参考点来自每回合自然稳定后的实测球心，而不是模型重叠初始化时的位置。使用 `link6` 坐标系。允许 X/Y/Z 分别偏移 8/15/10 mm，超过容许区后按 15/20/15 mm 的平滑尺度衰减。Y 朝手指方向。它是“掌心到手指根部的小范围支撑”，不是把球锁死在一点。

原地一直握着不增加奖励；前后摆动重复走同一段距离也不能重复得分。达到归一化距离后正常开手，新的滑动不再产生额外位置惩罚。没有强制关手、固定等待或规定动作轨迹。旧的 `arm_drive_*` 附加项在此配置关闭，避免重复计算。

## 初始化权重与验证

默认从 `BasketPhysicalArm_Continue/2026-10-28/18:10:13/nn/throw_basket_catch_best_reward_24.85.pth` 初始化网络与输入统计。这个权重已经适应无初始重叠、自然稳定后的球位置；其 24.85 分来自旧版折扣后的奖励，不能用数值判断它比 39 分模型差。

“原 39 分奖励函数”与“初始化网络权重”是两个选择。这里恢复前者，并使用用户实测 12/12 穿框的后者作为稳定起点。直接加载原 39.70 权重也通过了推理入口检查，但在自然稳定准备状态下有明显漏框，因此不设为默认来源。

只做了推理和代码检查，没有启动训练：

- 实际测试入口用新配置和 24.85 权重运行 12 回合：安全进框 12/12。
- 新公式平均回报约 40.44；这是公式恢复和附加项带来的分值，不是学习后进步。机械臂动作仍是同一权重的动作。
- 69 项相关测试通过，覆盖持球与滑球对照、奖励上限、原地持球和反复摆动不能刷分、一次释放给分、碰撞屏蔽以及实际环境保留完整 20 分进框奖励。
- 修复固定基座测试入口此前忽略 `test_num_episodes` 的问题；现在单环境设 30 回合会在完成后退出。

这只证明新奖励可计算且保留进框行为，不能证明训练后必然增加前送。正式训练要同时对比进框率、离手前前送距离、肩肘前向速度、离手时间和掌内位置质量。

## 用户手动启动训练

```bash
cd /home/isee/Catch/catch_it_copy
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 train_DCMM.py --config-name basket_aim_carry \
  test=False num_envs=32 vector_mode=async device_id=0 \
  checkpoint_catching=/home/isee/Catch/catch_it_copy/outputs/BasketPhysicalArm_Continue/2026-10-28/18:10:13/nn/throw_basket_catch_best_reward_24.85.pth \
  viewer=false basket_log=false basket_self_collision_debug=false \
  train.ppo.max_agent_steps=500000 output_name=BasketAimCarry
```

先以 50 万步作为观察窗口。加载网络与输入归一化，统计冻结；优化器、价值归一化与计步重新开始。学习率设为 `5e-5`。不自动启动训练。原模型与原配置保留。

训练后使用同一配置和新权重测试：

```bash
python3 train_DCMM.py --config-name basket_aim_carry \
  test=True num_envs=1 vector_mode=sync device_id=0 \
  checkpoint_catching=新权重完整路径 \
  viewer=true basket_viewer_speed=1.0 basket_log=true \
  train.ppo.test_num_episodes=30 output_name=BasketAimCarry_Test
```
