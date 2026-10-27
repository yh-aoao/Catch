# 先保留投掷，再适应移动投掷（2026-10-27）

现有`throw_basket_catch_best_reward_39.44.pth`作为固定目标投掷阶段的成果。
新配置`basket_throw_first`从该Actor初始化，复用用户已训练的停车Actor，
在真实完整移动回合与原固定目标回放回合间混合训练臂手。
不是从效果变差的`best_reward_-3.78`继续训练，也不重新训练已有停车模块。

## 复核到的问题

- 用户日志6秒时仍在navigate，停稳计时只有0.104秒；投掷未被开放。
- 原要求球/手每个物理子步连续接触0.30秒，短暂接触抖动也会清零停稳时间。
- 底盘已在位置区时仍不断跟随带观测噪声的策略速度指令，可能反复穿越速度门槛。
- 原第二阶段出手也要求底盘速度<0.06m/s；一次最后手接触时速度为0.1225m/s，故提前释放失败。
- 原第二阶段在导航期间屏蔽臂手动作，却仍用这些动作样本更新Actor；这些动作没有造成当步状态变化，增加无效更新和噪声。
- 导航与投掷共享8秒预算。停稳很晚时留给出手、自由飞行和安全观察的时间不足。
- 旧投掷模型原先开局即出手，移动托球数秒后的状态分布不同；只加载旧权重并不等于保留实际效果。

本机CPU复测用户第二阶段权重5回合：0次安全入框，2次未开放臂手即超时，
1次提前释放，2次合格抛出后落地。它是小样本诊断，不是200回合正式评估。

## 新配置的处理

- 到位置区后锁存零速度请求，越过1.5倍位置容差才重新开放导航请求；仍通过真实加速度限制/PID控制，不修改底盘位置或实际速度。
- 停稳阶段允许最多0.03秒的手接触短暂中断；无接触达到0.05秒的释放条件保持不变。
- 停稳进入条件仍为X/Y各0.07m、速度<0.06m/s、连续0.30秒。
- 已经真实停稳后，最后手接触时允许X/Y各0.10m、速度<0.15m/s，容忍出手反作用。
  未经历停稳仍不能合格释放；危险碰撞检测不变。这是仿真任务容差，不是硬件安全标准。
- 未到位的导航上限仍为8秒；到位后独立提供2.5秒投掷预算，并继续要求0.30秒命中后安全观察。
- PPO Actor损失、熵、bounds、KL及优势归一化只使用臂手已开放的样本。Critic仍学习完整回合。
  无有效动作样本的batch不更新Actor（包括Adam动量），也不因零KL提高学习率。
- 30%回合在重置时选原固定目标、从原始持球状态直接投掷；70%为原随机移动目标。
  没有在移动后传送球、重抓球或重置机械臂。回放回合也是独立的真实物理回合。
- 固定目标回放与移动入框成功率分开统计；best_success按移动回合成功率选，至少50个完整移动回合才开始选。

## 训练命令

```bash
conda activate dcmm
cd /home/isee/Catch/catch_it_copy
python3 train_DCMM.py --config-name basket_throw_first \
  test=False num_envs=32 vector_mode=async device_id=0 \
  basket_parking_checkpoint=/home/isee/Catch/catch_it_copy/outputs/BasketStage1_Parking/2026-10-27/19:23:21/nn/throw_basket_catch_best_reward_3.50.pth \
  basket_warm_start=/home/isee/Catch/10-25/throw_basket_catch_best_reward_39.44.pth \
  viewer=false basket_log=false basket_self_collision_debug=false \
  output_name=BasketThrowFirst
```

停车Actor保持冻结，臂手Actor允许适应移动后的状态；PPO仍为第二阶段18维动作更新。
这次优先修正训练接线与技能保留，不同时增加底盘学习自由度。
原`basket_stage2`保留，便于复现已有模型，不会默默改变其测试条件。

## 测试新模型

```bash
python3 train_DCMM.py --config-name basket_throw_first \
  test=True num_envs=1 vector_mode=sync device_id=0 \
  viewer=true basket_log=true basket_viewer_speed=1.0 \
  basket_mobile.fixed_throw_probability=0.0 \
  checkpoint_catching=/绝对路径/新训练模型.pth \
  train.ppo.test_num_episodes=200 output_name=BasketThrowFirst_MobileTest
```

测试时允许将回放概率改为0或1；其他控制和任务参数须与训练保持一致。
概率0测试纯移动入框，此时mean_success也是移动安全入框率。
混合回合统计时应看`mobile_success`和`rehearsal_success`，不能把总成功率当成移动成功率。
`basket_mobile/active_action_fraction`用于检查一批采样中臂手真正开放的样本比例。
日志新增`ready_time`、`deadline`、`rehearsal`，可判断延迟来自导航/停稳还是已开放后的投掷。

## 验证边界

- 87项相关测试通过，包含接触间隙、停稳锁存、分阶段预算、释放容差和Adam掩码更新。
- 原39.44模型在原固定目标诊断为10/10安全入框；新配置的纯回放诊断也为10/10。
- 迁移初始化但尚未训练适应的新策略，在5个纯移动诊断回合中4次开放臂手、0次入框。
  这说明已有技能在回放中保留，但移动后的状态适应尚未解决，不能宣称新训练已收敛。
- 真实旧权重与用户停车权重的异步短训练、保存和加载已跑通；混合短测试回放成功率1、移动成功率0，分组记录正常。
- 未启动长时间GPU训练，短训练正奖励主要可能来自回放，不作为移动效果改善的证据。
