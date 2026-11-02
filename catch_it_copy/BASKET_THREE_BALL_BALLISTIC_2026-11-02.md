# 三球混合 + 物理参数条件 + 改进弹道奖励

本版对应用户澄清的目标：保留单球独立验证，同时新增一版使用改进奖励的三球混合训练。没有启动训练。

## 默认训练命令

```bash
cd ~/Catch/catch_it_copy
bash scripts/train_basket_three_ball_ballistic.sh
```

默认32异步环境、GPU0、固定学习率3e-4、从零训练、1万亿步上限，可手动停止。

输出：`outputs/BasketThreeBallBallisticScratch/日期/时间/nn/`。

等价命令：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 train_DCMM.py --config-name basket_three_ball_ballistic \
 test=False num_envs=32 vector_mode=async device_id=0 \
 basket_mixed_source='' checkpoint_catching='' checkpoint_tracking='' basket_warm_start='' \
 viewer=false basket_log=false basket_self_collision_debug=false \
 train.ppo.max_agent_steps=1000000000000
```

`basket_three_ball_ballistic`是本次新奖励混合配置。旧`basket_three_ball`保留为旧奖励对照；先前的三个单球配置也保留。

## 根据球的差异学习

每回合等概率选原始球、网球、棒球，回合内保持同一个球，底座固定。单一共享策略学习：

`动作 = policy(机器人状态、球的位置/速度、球半径、球质量)`。

第一版半径和质量来自仿真真值，输入实际单位分别为米和千克。网络前处理为`(radius-0.04)/0.01`与`log(mass/0.05)`。球类ID仅用于采样/成功率/损失分组，不是策略输入；没有RGB或外貌识别。实际球质量、半径、惯量和摩擦在环境内随球改变，因此相同控制产生的球运动也不同。

策略控制机械臂与手指的动作，不直接输出一个“力值”；通过加速过程、手腕/手指配合及释放时机形成不同的出手速度和方向。它可以根据条件输入产生不同动作，不要求三种球共用一个折中动作。是否学到合理差异必须看训练后各球的独立测试，不能由输入接通就断言已学会。

当前只显式输入半径、质量；惯量/摩擦在这三种已知球中按各自物理先验设置，尚未作为独立观测输入。若以后想泛化到同尺寸同质量但不同材质的球，需要进一步提供或估计相应动力学信息。

## 感知接入边界

之后可以在`basket_ball_properties`接口提供估计的半径/质量，单位和顺序保持不变。当前代码没有实现视觉测量、质量估计或感知噪声随机化；不能把真值条件下的成绩解释为自动识别未知球的能力。质量可由称重、力觉或运动响应估计，单靠静态外观通常不足以唯一确定。

## 改进奖励与对照

本版启用与单球实验相同的ballistic_v2：

- 持球加速时，根据最后可能释放状态预测框平面误差，仅奖励本回合新的最好弹道质量，进展预算8，重复摆动不反复支付。
- 有效抛出从5缩为2，飞行推进上限从5缩为3。
- 机械臂前送、位置保持、前送释放上限分别2/1/2，并乘弹道质量，减少独立短抛奖励。
- 未投进的终止回合额外-8一次；原无效抛出、碰撞及手指协同惩罚保留。
- 原弹道瞄准最高10与真实投进20保留，篮筐、安全和有效释放条件不放宽。

三球损失按整份rollout等权平衡，各球分别标准化优势。网络、critic、归一化均从零开始，不加载41.30、12.48或移动模型，不使用旧策略参考损失。

共享奖励配置为`basket_ballistic.yaml`，三球混合与三个单球配置都从它继承。因此可以比较相同奖励下单球是否比混合更容易学习，而不把奖励差异混入比较。

## 测试新三球模型

```bash
python3 train_DCMM.py --config-name basket_three_ball_ballistic \
 test=True num_envs=1 vector_mode=sync device_id=0 \
 checkpoint_catching=/完整路径/新三球输出/nn/throw_basket_catch_best_success.pth \
 basket_mixed_source='' viewer=true basket_viewer_speed=1.0 basket_log=true \
 train.ppo.test_num_episodes=300
```

公平测试分别设置`basket_mixed_ball=original`、`tennis_ball`、`baseball`各100回合。新奖励配置不能静默加载旧12.48奖励模型。日志分类SR、Macro SR及Actor updates可用来观察各球与实际PPO更新情况。

## 验证与同步

23项相关测试通过，包括三球/单球奖励一致、无旧权重路径、等概率采样配置、不同物理参数到actor的输入链路，以及原奖励与环境回归。两个spawn环境采样到了ID 0/1/2全部三球，验证37维观测/18维动作、ballistic_v2、无源权重初始化、元数据存储、保存重载与有限推理步。未执行优化器更新，也没有训练成绩。

本次新增/调整：

- configs/basket_ballistic.yaml（新增共享奖励配置）
- configs/basket_three_ball_ballistic.yaml（新增三球混合配置）
- configs/basket_single_delivery.yaml（改为继承共享配置）
- scripts/train_basket_three_ball_ballistic.sh（新增）
- tests/test_basket_ballistic_configs.py（新增）

需同时同步上次ballistic_v2的环境/奖励/训练器代码，详见BASKET_SINGLE_BALL_2026-11-02.md。诊断记录：`/home/isee/Catch/analysis_artifacts/basket_three_ball_ballistic_20261102/`。
