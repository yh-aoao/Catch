# 固定底座单阶段投掷基线

2026-09-21：用户已通过 viewer 验证当前初始化能自然持球。

## 行为与范围

- `--config-name=basket_fixed` 选择 Catching_OneStage、throw_basket、固定目标及底座零速度控制。
- 策略输出 18 维：6 维末端位姿增量，经 IK 控制机械臂；12 维手指控制。底座两维在执行时补零，不进入策略分布。
- 这是底座零速度伺服，不是焊死底盘；仿真接触反作用可能引起小幅漂移。
- 篮筐使用 DcmmCfg.basket_center，横向固定为 0；球参数及物理初始化沿用当前 Basket。原有 PID、动作延迟等随机化仍保留，不宣称所有动力学完全固定。
- 保留当前 joint_throw_reward 和真实穿框判定，先隔离训练结构，不同时更换奖励。没有额外托球、吸附或球速注入。
- 单阶段从第一步联合学习臂手，不加载 Track。旧 20 维 OneStage、TwoStage 模型不可直接加载；此基线严格检查模型形状。
- 训练与测试统计通过终止步 final_info 读取成功，避免自动 reset 后丢失成功标志。

## 训练（仓库 catch_it_copy 根目录）

```bash
python3 train_DCMM.py --config-name=basket_fixed test=False num_envs=32 viewer=false imshow_cam=false
```

先短跑可追加 `train.ppo.max_agent_steps=100000 basket_log=true`；长训练关闭逐回合日志。GPU 用 `device_id=1` 等选择。

## 测试

```bash
python3 train_DCMM.py --config-name=basket_fixed test=True num_envs=1 checkpoint_catching=/absolute/path/to/new_model.pth viewer=true imshow_cam=false basket_log=true
```

不要传 checkpoint_tracking，也不要用旧的 TwoStage Catch 模型。测试保留同一 config-name。

输出位于 outputs/BasketFixedOneStage/日期/时间/，包含 resolved_config.yaml，模型位于 nn/。检查点保存 action_layout。Python 环境参数仍由 configs/env/DcmmCfg.py 控制，复现实验需同时保留对应代码版本。

优先观察：球是否持留到加速、释放时速度、有效脱手率、真实入框率及失败原因；不能仅根据 reward 上升断言学会抛球。

本次已验证动作切片和严格检查点加载，Python 语法检查通过。本机 Python 缺少 hydra，尚未执行完整配置组合及训练入口；实际收敛和底座漂移需服务器物理仿真验证。
