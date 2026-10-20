# Basket direct launch v5

针对下摆后掉球而没有有效前抛的问题，新增显式选择的 launch_v5。
保留静态持球的初始姿态，不添加脚本下摆、定时开手、球速度或外力。
低位新姿态尚未经过仿真验证，因此本轮不修改初始关节角。

- 接触且满足原工作空间和释放距离约束时，前向、上向加速分别获得最多25%的进展奖励，协同加速最多50%。总加速奖励仍最多6分；重接触不能重置预算。
- 增加球在link6运动坐标系中的滑动速度反馈，接触期间低滑动获得有限承托奖励，每回合最多1分。它是运动稳定性近似，不是精确掌心区域判定。
- 接触期间球向下速度超过0.1m/s，按超出速度乘仿真时间给予轻微惩罚；自由飞行不受此项惩罚，不硬禁止下摆。
- 有效释放仍要求前向、向上速度均至少0.5m/s，至少自由飞行0.1秒且前进0.15m，获得一次10分奖励；入框仍用原几何判定和实际飞行约束。
- 日志新增held_slip_speed和held_downward，reward_version显示launch_v5。
- 默认legacy与launch_v4保持原奖励路径；Throw/Roll/Bounce不切换到v5。

从头训练固定底座单阶段：

```bash
python3 train_DCMM.py test=False task=Catching_OneStage num_envs=32 \
  object_motion=throw_basket basket_fixed_base_training=true \
  basket_arm_control=joint_velocity basket_reward_version=launch_v5 \
  basket_curriculum_stage=0 basket_log=false \
  device_id=0 output_name=BasketLaunchV5Stage0
```

先看有效抛出比例和释放速度，再用checkpoint_catching加载权重逐步将课程改为1、2、3。
原PPO的加载是权重初始化继续学习，不是完整优化器状态恢复。
正式测试用最终距离stage3：

```bash
python3 train_DCMM.py test=True task=Catching_OneStage num_envs=1 \
  object_motion=throw_basket basket_fixed_base_training=true \
  basket_arm_control=joint_velocity basket_reward_version=launch_v5 \
  basket_curriculum_stage=3 basket_log=true viewer=true \
  checkpoint_catching=/absolute/path/to/model.pth
```

测试老模型不会让它自动学到新行为，建议新输出目录从头训练。
本地测试不能代替服务器真实物理验证，尤其需要确认动作能实际产生前上方加速。
