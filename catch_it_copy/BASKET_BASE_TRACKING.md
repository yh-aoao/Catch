# Basket 第一阶段：底座站位

Tracking 只输出底座 x/y 两个速度动作。机械臂通过 PID 保持 `DcmmCfg.arm_joints`，
手指保持杯状，球仍由原持球机制放在手中。

世界 X 是左右，Y 是前后。站位目标使用 arm_base 的世界坐标：
`[basket_x, basket_y - basket_base_front_dist]`。
当前篮筐中心 y=2.2m、站位距离=0.6m，因此目标 y=1.6m；x 随篮筐中心变化。
配置在 `configs/env/DcmmCfg.py` 的 `basket_base_front_dist`。
这里的目标是 `arm_base`，它与底盘几何中心有安装偏移，不应以车壳中心判断是否到位。

停车任务使用独立奖励：负距离、有正负号的靠近增量、速度偏差惩罚、
底座动作惩罚和时间代价；成功额外 +100，提前失败额外 -100。
速度参考为目标误差乘 1.5、合速度封顶 0.8m/s，进入目标区后参考速度为零。
它仅用作奖励，机器人实际速度仍由策略输出；没有直接修改底座位置或自动驶向目标。
相同位置下，仍高速移动会比停稳获得更低奖励，越过目标后参考方向会反向。

成功要求世界 x/y 各误差小于 0.08m、底座平移速度小于 0.10m/s，连续满足 5 个策略步
（默认约 0.2s）；中间离开目标区或速度超限就清零。回合时限仍是 4 秒。
速度范数限制为 0.8m/s，两阶段采用相同限制。对应参数均为 `basket_track_*`。

历史 best_reward=0 的一个确定代码问题：首个回合尚未结束时，空统计器返回 0，
旧代码会把它存为 best；原来全负奖励无法超过它。现在只有统计到完整回合才记录
回合指标或选择 best，负回报也可以正常成为首个 best。`last` 仍正常保存。

在 catch_it_copy 目录训练：

```bash
python3 train_DCMM.py test=False task=Tracking num_envs=32 object_motion=throw_basket output_name=BasketParking
```

测试新模型：

```bash
python3 train_DCMM.py test=True task=Tracking num_envs=1 viewer=True object_motion=throw_basket checkpoint_tracking=/path/to/new_track.pth basket_log=true
```

第二阶段模型动作分工为冻结底座分支 2 维 + 可训练臂手分支 18 维。
同时修正了两阶段篮筐观测拼接顺序，确保 tracking 输入一致、最后 12 维为手关节。
保留 `basket.rel_pos3d`（篮筐中心相对 arm_base 的世界坐标差），新增
`basket.target_rel_pos2d`（停车目标在车体坐标系中的 XY 误差），让停车目标、
底座速度和动作方向一致，并保留第二阶段抛球所需的篮筐信息。
因此 basket Tracking 观测为 23 维、动作 2 维；Catching 观测为 35 维、动作 20 维。
旧 basket 模型需要重新训练；Tracking 加载时拒绝不兼容权重，不再裁剪加载。

`basket_log=true` 每 25 步及回合结束打印 `[basket-track]`：目标位置、世界 XY 误差、
实际速度、连续停稳计数、每步 reward、各分项和结束原因。可调整
`basket_track_log_interval`。正式 32 环境训练建议保持日志关闭，避免大量输出。
奖励为负并不等于没有学习，应同时观察误差是否下降、是否停稳和成功率。

Linux 上请同步本次全部代码（包括新增 `gym_dcmm/utils/basket_tracking.py`、环境、
配置、训练入口和三个 PPO 文件），然后新建训练；不要只同步奖励配置。

第二阶段现有按时间释放球的机制仍在使用；本次没有把它改为“站稳后才释放”。
CPU 测试不代表已验证仿真收敛或实机可达性。
