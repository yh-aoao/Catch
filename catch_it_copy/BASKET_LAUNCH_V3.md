# 固定底座主动投掷 v3

## 改动原因与范围

原Basket使用末端笛卡尔增量IK：动作反归一化为0.025，策略周期通常0.04秒，目标平移尺度约0.625m/s（不是严格的真实末端速度上限）。而此前日志中所需出手速度约3m/s水平、3m/s向上。仅反复增加命中奖励不能排除控制尺度不足的问题。

原来`velocity_progress`主要取弹道质量，不是直接奖励早期持球加速。因此新增可选择的关节速度动作语义及物理子步加速奖励。它是待训练的新基线，不是已证明解决投掷的问题。

## 新控制模式

设置`basket_arm_control=joint_velocity`，当前仅支持`basket_fixed_base_training=true`和`task=Catching_OneStage`。

机械臂6维动作归一化后映射为关节速度目标，速度尺度[2,3,3,3,3,3]rad/s，再转为当前关节位置加上速度×策略周期的PID目标。限幅到关节范围，沿用现有执行器/PID；真实速度仍需日志验证。避免依赖多次小幅末端平移来探索投掷。

手指12维动作由策略直接控制，保留Catch原有0.15rad动作尺度，不额外放大；新增目标关节范围限幅。释放由真实接触消失、距离和去抖时间判定，无定时开手、无给球添加速度、无粘球。

默认`basket_arm_control=cartesian`保持历史行为。新旧动作语义不同，须从头训练。固定底座单阶段checkpoint记录控制模式；把旧笛卡尔模型放进新模式测试会明确报错。

## 新奖励和判定

- 保留原弹道质量进展、有效释放、实际入框奖励和失败代价。
- 新增`held_acceleration`：只在真实手接触且未递到篮筐附近、机械臂未过度前伸时，比较向目标的水平速度和向上速度相对于参考出手速度的完成比例。
- 使用物理子步采样，避免高速出手发生在两个策略步之间而漏掉加速过程。
- 仅奖励本回合新的最高完成比例，整回合该项最多6分；重新接触不重置预算，反复抖动不能无限刷分。
- 新模式有效出手需满足已有的最小距离0.8m，以及水平朝目标、向上速度分别至少0.5m/s。这些只是排除递球/掉球的门槛，不是最终目标速度。
- 得分仍需物体真正自由飞行并穿过篮筐开口，保留最短飞行时间和水平飞行距离要求。不能仅凭出手快就判成功。

## 从头训练与测试

从`catch_it_copy`目录运行：

```bash
python3 train_DCMM.py test=False task=Catching_OneStage num_envs=32 \
  object_motion=throw_basket basket_fixed_base_training=true \
  basket_arm_control=joint_velocity basket_log=false \
  checkpoint_tracking='' checkpoint_catching='' \
  device_id=0 output_name=BasketJointLaunchV3
```

测试：

```bash
python3 train_DCMM.py test=True task=Catching_OneStage num_envs=1 \
  object_motion=throw_basket basket_fixed_base_training=true \
  basket_arm_control=joint_velocity basket_log=true \
  checkpoint_catching=/absolute/path/to/new_basket_v3.pth \
  viewer=true object_eval=false device_id=0
```

将模型路径替换为本次新训练模型；无界面测试改`viewer=false`。不要用`basket_throw_probe.py`的脚本摆臂结果当作新策略结果，它仍是旧控制链的诊断工具。

优先看`held_acceleration`是否出现、真实`arm_joint_speed`/`ball_velocity`是否增长、`contact_total_seconds`、出手快照与`throw_valid`，再看入框成功率。如果加速前就掉球，应先解决初始握持/动作探索；如果关节目标误差很大而真实速度很小，继续检查执行器限幅与PID。不要直接提高力量奖励或放松入框标准。

先验证固定目标、固定底座专家，再扩展目标距离课程和移动底座。当前没有自动课程，也没有声称已训练成功。

## 验证与兼容性

Basket抓取/投掷22项、控制11项、固定策略3项测试通过；新增斜坡/速度映射/奖励预算/动作语义测试另见`tests/test_slide_and_launch.py`。旧base-only Tracking测试中3项在修改前HEAD上同样失败，本轮未更改历史测试断言掩盖它们。

本机MuJoCo DLL不可加载，真实环境速度、碰撞和学习效果仍待服务器验证。新增参数不改变默认Cartesian模式的投掷奖励与控制；Throw/Roll/Bounce没有切换到新逻辑。

注意：蒸馏程序会记录源代码哈希。本轮更改共享环境源码后，不要在正在训练的蒸馏工作目录直接更新代码；保留原训练代码版本供该模型恢复和评估，在独立目录验证新运动任务。
