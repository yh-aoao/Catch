# 固定三球 Catch → 参数随机化微调

新实验直接加载完整固定参数三球 Catch，不重新训练 Track，也不随机初始化手指策略。

默认来源：

- Roll：`11-5/roll_catch_best_reward_270.37.pth`
- Bounce：`11-5/bounce_catch_best_reward_13.68.pth`

```bash
conda activate dcmm
cd /home/isee/Catch/catch_it_copy
bash scripts/finetune_three_ball_catch_parameter_v1.sh roll dr device_id=0
bash scripts/finetune_three_ball_catch_parameter_v1.sh bounce dr device_id=0
```

可用 `checkpoint_catching=/绝对路径/source.pth` 替换来源。原文件不会修改。

## 保留和重置的内容

完整 Track/Catch actor、critic，以及 Track/手部/价值归一化统计严格加载。
新建优化器，重置训练步数、轮次、最佳奖励与随机数进度。这是新分布上的新实验，
不是恢复旧训练，也不会继承 Bounce 旧最佳奖励阈值而导致新 best 难以保存。

配置开关 `catch_warm_start=true` 只由新入口显式启用，默认 false；旧命令加载逻辑不变。
微调学习率为 **1e-4**（原基础训练默认5e-4），Track/Catch联合优化，不冻结。
原始球、网球、棒球均分，24环境，每球8套物理参数；默认seed123，最大新增步数上限10亿。
改变num_envs时自动计算minibatch。奖励与当前三球Catch训练版本保持一致。

## 物理随机化

复用physical_dr_v1：尺寸±5%、质量±10%、运动表面滑动摩擦±15%；
Roll滚动阻力±20%；Bounce接触阻尼比±10%。尺寸、惯量、碰撞与视觉同步调整。
每环境创建时采样一次，之后固定；每回合随机发射轨迹。不是每回合重采样物理参数。
没有额外的扰动幅度课程或基准参数混入，避免本次同时引入其他实验变量。

输出：

- `outputs/roll_three_ball_catching_parameter_dr_from_fixed_v1_seed123/`
- `outputs/bounce_three_ball_catching_parameter_dr_from_fixed_v1_seed123/`

输出provenance记录来源模型绝对路径、SHA256、采样参数与相关源码摘要。

## 同源固定参数对照

将dr换成fixed，可从同一Catch起点、同一学习率继续固定参数训练：

```bash
bash scripts/finetune_three_ball_catch_parameter_v1.sh roll fixed device_id=0
bash scripts/finetune_three_ball_catch_parameter_v1.sh bounce fixed device_id=0
```

输出目录将dr替换成fixed。比较相同新增步数，两组都测试基准参数和未见过随机样本，
判断是否保留基准能力并提高泛化。成功率需实际训练后验证，不能由初始化方式保证。

## 继续中断的微调

本入口默认每次都开始新微调实验。Bounce若恢复本次微调保存的last，应显式覆盖
`catch_warm_start=false checkpoint_catching=/本次输出/bounce_catch_last.pth`，恢复优化器和步数。
Roll现有格式不含优化器/步数，仍只能加载权重继续学习。

旧源码备份：`backups/before_catch_warm_start_20261109/`。
