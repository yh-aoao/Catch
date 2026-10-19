# Basket launch_v4

新增可选版本，不改变默认legacy/v3。仅用于固定底座、关节速度控制、Catching_OneStage。

## 改动

持球加速改为向前/向上完成比例的几何平均，乘侧向速度惩罚。保留回合新最高值奖励上限，纯下落或纯向上不能获得完整加速奖励。

有效抛出与投准分开：出手满足已有最小距离、向前和向上各0.5m/s条件后，至少自由飞行0.1秒且朝目标前进0.15m，给一次10分。任何再次接触使当前候选失效，包括碰手、机器人、地面或篮筐；已获得的回合奖励不会重复发放。飞行时间从最后接触快照计起。此奖励不要求入框，掉球不符合出手速度条件。

原valid_release弹道质量奖励改名throw_accuracy；实际入框成功判定不变。终端basket_control增加throw_confirmation（原因、飞行时间、前进距离、出手速度）与curriculum_stage。

可调参数在DcmmCfg.py：basket_v4_throw_reward、basket_v4_flight_seconds、basket_v4_flight_travel。

## 课程与运行

本版实现手动距离课程：stage 0/1/2/3篮筐世界Y为1.5/1.7/1.9/原配置2.2m，篮筐大小、Z高度和成功标准不变。不是自动升级，也没有实现宽目标区域；先验证真实抛出，避免同时改太多判定。近目标仍须满足最小释放距离，不能递到框里。

从头训练：

```bash
python3 train_DCMM.py test=False task=Catching_OneStage num_envs=32 object_motion=throw_basket basket_fixed_base_training=true basket_arm_control=joint_velocity basket_reward_version=launch_v4 basket_curriculum_stage=0 basket_log=false device_id=0 output_name=BasketLaunchV4Stage0
```

阶段完成后用checkpoint_catching加载该阶段模型，改stage=1、2、3及output_name。现有PPO加载属于权重初始化继续训练，不是完整优化器/随机数状态断点续训。阶段升级先人工检查有效抛出和当前入框表现，不按训练轮数盲目升级。模型动作与v3兼容，但推荐新基线从头训练。

正式测试强制stage=3，避免近距离成绩当最终任务成绩：

```bash
python3 train_DCMM.py test=True task=Catching_OneStage num_envs=1 object_motion=throw_basket basket_fixed_base_training=true basket_arm_control=joint_velocity basket_reward_version=launch_v4 basket_curriculum_stage=3 basket_log=true viewer=true checkpoint_catching=/absolute/path/to/model.pth
```

观察近距离训练过程可用训练命令num_envs=1 viewer=true basket_log=true做短暂诊断。正式测试不能指定早期课程阶段。

## 验证

新增3项方向/飞行/掉落测试通过，原Basket22项及Slide/控制兼容6项通过，语法检查通过。本机MuJoCo DLL限制导致无法真实运行，奖励是否足够、近目标空间和控制器实际速度仍需服务器验证。不修改Throw/Roll/Bounce，不添加球外力、定时开手或脚本摆臂。
