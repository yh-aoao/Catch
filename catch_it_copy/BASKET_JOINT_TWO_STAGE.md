# Basket 两阶段联合投掷

## 新奖励（覆盖旧 Catch 奖励组合）

实际入口 joint_throw_reward，供两个训练阶段共用。

- 每策略步时间成本 -0.05；真实入框 +100；失败或超时 -20，终局奖励仅一次。
- 持球准备：q=clip(1-||v-v_ref||/||v_ref||,0,1)，仅超过本 episode 历史最佳 q 的增量获得 8*增量奖励，准备进度累计最多 8。不可行参考速度不给奖励。
- 承托总奖励最多 0.2（0.2 秒额度乘 1），不再因移动准备超过 0.2 秒惩罚持球。
- 有效释放奖励为 8*释放质量，每回合一次；保留物理释放和质量门槛。取消固定释放加分及额外向前速度奖励的重复累加。
- 飞行奖励保留有符号的到框距离改善；远离会扣分，静止无奖励。最终仍以真实过框判定。
- Track 仅惩罚底座/机械臂动作均方和动作变化；Catch 额外约束手指。没有张手、闭合或动作幅度的正奖励，由投掷与释放效果引导手指学习。
- 日志新增 totals，显示本回合各项奖励累计，便于区分终止罚分、进度奖励和控制成本。

没有新增强制停车位置奖励；底座位置通过投掷结果联合优化。固定手姿的物理可行性仍需 viewer 验证。新奖励改变回报尺度，旧 best reward 不能直接横向比较。

本记录取代旧的“Track 仅停车，Catch 再抛球”方案。

- Tracking：8 维动作（底座 2 + 机械臂 6），手指维持 reset 的杯状姿态。
- Catching_TwoStage：继承新的 Tracking，20 维动作（底座 2 + 机械臂 6 + 手指 12），全部联合学习；8 维跟踪分支不会触发旧的底座冻结选项。
- 两阶段都直接进入 preparing，采用投掷速度质量、接触加速、释放、飞行和入框奖励；成功必须真实过框，不再以停车成功结束。
- 球仅在 reset 放到手旁，后续无位置锁定、重力补偿或发射速度赋值。底座和机械臂全程可动作。固定杯状姿态能否稳定托球并自然释放尚需仿真验证。
- 原 2 维底座 Track 及对应 2+18 动作拆分的 Catch checkpoint 不兼容，必须重新训练。

```bash
python3 train_DCMM.py test=False task=Tracking num_envs=32 object_motion=throw_basket basket_log=true
python3 train_DCMM.py test=False task=Catching_TwoStage num_envs=32 object_motion=throw_basket checkpoint_tracking=/path/to/new_8_action_track.pth basket_log=true
```

日志为 `[basket-throw] task=Tracking/Catching ...`，检查实际球速、参考速度、接触和 IK 成功次数。当前保留原篮筐范围、时限和速度限制，未声称已经验证学习收敛。

验证：14 项 Basket 单元/回归测试（含无辅助、联合策略梯度测试）通过，Python 语法检查通过。未进行 MuJoCo 长程训练。
