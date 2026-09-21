# Basket 两阶段联合投掷

## 44 ms 脱手诊断更新

准备到飞行切换需连续无接触 0.03 秒且满足原离掌距离条件，中途重新接触会重置计时。释放弹道质量、出手位置和时间取最后接触快照，避免确认延迟中的重力变化使原本向上出手被误判。没有把球重新绑定到手上。初始化与奖励权重保持不变，真实承托可行性尚未验证。

新增仅 test=True 可用的 basket_control_probe：hold 固定底座/臂/手目标，arm 固定底座和手仅允许策略控臂，hand 固定底座和臂仅允许策略控手，off 正常控制。固定指的是保持目标指令，关节仍受物理和 PID 控制。使用原测试命令及对应 Catch checkpoint，追加 basket_control_probe=hold basket_log=true，随后对照 arm 和 hand。探针统计不能作为正常策略性能报告。

日志末尾包含 probe、contact_total_seconds、no_contact_seconds。先比较静止托球和策略动作对照，再决定是否改球初始位置；不凭 44 ms 日志直接猜新的托球偏移。

## 远处平面穿越与释放诊断修正

球在圆框开口外穿过篮筐延伸平面不再立即终止。保留地面/距离越界/超时及真实入框判定；进入开口但不满足有效投掷仍为 not_a_throw。奖励权重不变，但回合可能更长，累计奖励不能直接与旧终止逻辑比较。

开启 basket_log=true 后，脱手检测时立即输出一次 basket-release：检测时刻、位置、速度、参考速度、水平距离、累计实际手球接触秒数，以及最后一次接触的位置/速度和连续接触时长。快照也在 info['basket_release'] 中，后续仿真不会改写。

注意检测释放仍包含离掌距离条件，因此检测时刻不一定恰好是失去接触瞬间；last_contact 单独记录最后接触状态，便于核对。接触时长来自物理子步而非 support 奖励，不受站位奖励门控影响。实际托球能力仍需运行仿真验证，本次不改初始化和控制参数。

## 2026-09-20：防止贴框放球

默认出手点到框中心水平距离至少 0.8 m；过框前自由飞行至少 0.15 s、从出手点水平位移至少 0.5 m。飞行期间再次接触手使该次投掷失效。贴框放入记为 not_a_throw，不获得入框成功奖励。持球准备期间，小于最小出手距离产生逐步增大的 too_close 惩罚，最大每步 2。

配置位于 DcmmCfg.py：basket_min_release_distance、basket_min_flight_time、basket_min_flight_travel、basket_near_hoop_cost。日志增加 throw_valid 和 horizontal_distance。阈值是初始实验设置，仍需实际训练验证。

同次 Roll Catch 调整：roll_grasp_open_target 从 0.15 改为 0.45 rad，reset 实际关节和控制目标同步预弯，仅修改 12 个可控屈曲关节；掌内目标仍 0.8 rad。Roll Tracking 初始姿态保持原样。

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
