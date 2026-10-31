# Basket 手指协同微调

新增配置 `basket_finger_sync`，从已验证的 38.38 权重初始化。沿用 `basket_aim_carry` 的完整进框奖励、机械臂前送、持球位置奖励，以及无重叠放球和自然稳定准备。观测仍为 35 维，动作仍为机械臂 6＋手指 12 维，无需改变原权重动作含义。

## 五项新增惩罚

| 日志项 | 默认权重 | 定义 |
|---|---:|---|
| `finger_sync` | 2.0 /s | 比较三根长指对应关节的归一化开合进度，差异超过 0.20 后平滑扣分。 |
| `finger_coupling` | 1.0 /s | 同一长指三个关节的开合进度偏离该指均值超过 0.25 后扣分，允许正常的屈伸先后差异。 |
| `finger_extension` | 2.0 /s | 对实测长指低于自然张开位 0.05 rad 的过伸，以及十二个关节接近／超出模型软限位的程度扣分；拇指不使用长指的负角度阈值。 |
| `finger_posture` | 1.0 /s | 有效发射确认后，用 0.15 秒逐渐引导到自然张开位；三长指和拇指各用自己的角度尺度，不做指间拇指同步。 |
| `finger_smoothness` | 0.01 /策略步 | 十二个归一化手指命令相邻步变化的均方惩罚；回合开始的前一命令为零。 |

开合进度的参考托球角取每次自然稳定准备完成后的实测关节角，自然张开角由配置给定，默认三长指与拇指各为零位。三根长指比较对应关节的相位，不要求各关节原始角度相等。拇指不纳入三指同步或指内耦合。

前四项在 2 ms 物理步测量真实关节状态，乘物理步长后累积，避免随采样频率改变权重。各误差先平方，再通过 `z/(1+z)` 有界平滑映射，严重分叉时仍保留误差改善信号。动作平滑项在每个策略步支付一次。

这些都是软惩罚，不投影／覆盖控制命令，不强制固定手型或规定固定开手时刻。持球和释放过程中没有离手后姿态惩罚；确认发射后该项才渐进启用。原 20 分进框奖励和安全判定保持原逻辑。

每个新权重保存 `aim_carry_fingers_v1` 奖励协议和具体参数，防止后续误用旧奖励配置测试。旧的 38.38 模型可直接作为初始化权重，新模型应使用同一个 `basket_finger_sync` 配置测试。

## 验证

- 80 项相关测试通过，涵盖指间协同、单指反折、拇指独立、实测越限、时间步无关性、释放渐变、十二通道平滑、检查点兼容和原有 Basket／球体测试。
- 使用 38.38 原权重、CPU、无界面、单环境、默认种子完成 12 回合纯推理检查：12/12 安全进框。
- 五项新增惩罚平均每回合合计约 −1.062 分，其中同步 −0.537、耦合 −0.125、过伸 −0.060、离手后姿态 −0.221、动作平滑 −0.119。
- 这是旧策略在新评分下的入口和量级检查，没有更新权重，不能据此声称手型已经改善。
- 未启动正式训练，也未启动短训练。

## 用户手动启动训练

```bash
cd /home/isee/Catch/catch_it_copy
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 train_DCMM.py --config-name basket_finger_sync \
  test=False num_envs=32 vector_mode=async device_id=0 \
  checkpoint_catching=/home/isee/Catch/catch_it_copy/outputs/BasketAimCarry/2026-10-29/01:23:05/nn/throw_basket_catch_best_reward_38.38.pth \
  viewer=false basket_log=false basket_self_collision_debug=false \
  train.ppo.max_agent_steps=500000 output_name=BasketFingerSync
```

学习率为 `3e-5`，输入归一化统计冻结，所有机械臂与手指策略参数仍参与训练。加载原网络和输入统计；优化器、训练计数与价值归一化重新开始。输出保存到新目录，不覆盖原模型。

训练后测试：

```bash
python3 train_DCMM.py --config-name basket_finger_sync \
  test=True num_envs=1 vector_mode=sync device_id=0 \
  checkpoint_catching=新权重完整路径 \
  viewer=true basket_viewer_speed=0.25 basket_log=true \
  train.ppo.test_num_episodes=30 output_name=BasketFingerSync_Test
```

同时比较安全进框率与五项手指惩罚；不能只用总分选择新模型。特别检查离手前是否保持有效支撑，离手后是否逐渐自然张开、是否仍有单指反向折叠。
