# Basket 第一阶段：底座站位

Tracking 只输出底座 x/y 两个速度动作。机械臂通过 PID 保持 `DcmmCfg.arm_joints`，
手指保持杯状，球仍由原持球机制放在手中。

世界 X 是左右，Y 是前后。站位目标使用 arm_base 的世界坐标：
`[basket_x, basket_y - basket_base_front_dist]`。
当前篮筐中心 y=2.2m、站位距离=0.6m，因此目标 y=1.6m；x 随篮筐中心变化。
配置在 `configs/env/DcmmCfg.py` 的 `basket_base_front_dist`。

距离奖励在整个移动范围内有效，另保留底座控制惩罚和碰撞/限位惩罚。
成功要求 x/y 各误差小于 0.15m，底座平移速度小于 0.15m/s。

在 catch_it_copy 目录训练：

```bash
python3 train_DCMM.py test=False task=Tracking num_envs=32 object_motion=throw_basket
```

测试新模型：

```bash
python3 train_DCMM.py test=True task=Tracking num_envs=1 viewer=True object_motion=throw_basket checkpoint_tracking=/path/to/new_track.pth
```

第二阶段模型动作分工为冻结底座分支 2 维 + 可训练臂手分支 18 维。
同时修正了两阶段篮筐观测拼接顺序，确保 tracking 输入一致、最后 12 维为手关节。
篮筐观测改为中心相对 arm_base 的世界坐标差。
旧 basket 模型的动作维度或观测语义不同，需要重新训练，不能直接续训。

第二阶段现有按时间释放球的机制仍在使用；本次没有把它改为“站稳后才释放”。
CPU 测试不代表已验证仿真收敛或实机可达性。
