# 肩肘参与前送的训练配置

## 更新：以已验证的 39.44＋新手型为正式训练起点

用户确认原 39.44 权重配合新手型更稳定，因此当前推荐 `basket_fixed_arm`，而不是下面早先的 `basket_hold_arm` 方案。新配置保留原模型的固定基座命令、原球生成方式、原目标、原启动时序、35 维观测与 18 维动作，不引入等待持球和 51 维观测。新手型为 `[0.8,0.7,0.7]`。

加载原 39.44 模型和原输入归一化，冻结输入统计，学习率降为 `1e-4`，新增同样的有上限前送奖励（进度最多 2、有效释放最多 2）。原抛球与进框奖励、碰撞惩罚继续生效。输入归一化冻结不等于冻结机械臂：策略所有关节仍参与优化。

```bash
cd /home/isee/Catch/catch_it_copy
python3 train_DCMM.py --config-name basket_fixed_arm \
  test=False num_envs=32 vector_mode=async device_id=0 \
  checkpoint_catching=/home/isee/Catch/10-25/throw_basket_catch_best_reward_39.44.pth \
  viewer=false basket_log=false basket_self_collision_debug=false \
  output_name=Basket39_ArmDrive
```

配置默认训练 200 万步。此入口使用 `checkpoint_catching` 加载原网络与输入统计；优化器和训练步数重新开始，不是精确恢复原训练进度。不要使用仅适用于移动任务的 `basket_warm_start`，也不需要测试专用的 `basket_eval_hand_flex` 开关。

新模型测试：

```bash
python3 train_DCMM.py --config-name basket_fixed_arm \
  test=True num_envs=1 vector_mode=sync device_id=0 \
  checkpoint_catching=新权重完整路径 \
  viewer=true basket_viewer_speed=1.0 basket_log=true \
  train.ppo.test_num_episodes=30 output_name=Basket39_ArmDrive_Test
```

双环境 CPU 短训练已完成，保存结果与原模型相比，输入归一化均值完全一致，actor 第一层参数最大变化约 `9.1e-4`，确认使用了原网络且发生了更新。短训练仅用于检查执行链路，不代表性能提高。不同奖励版本下的总分不能直接与 39.44 比较，应看进框和安全指标。

## 以下为早先的延迟持球方案记录

新增 `basket_hold_arm`，继承 `basket_hold_grip` 的持球姿势。动作维度、关节速度限制和开手动作语义保持一致，不施加额外助抛力，也不自动执行固定前送轨迹。

原策略已能控制六个机械臂关节，但奖励主要评价球的飞行和进框，不区分肩肘输送与末端动作。新配置增加两项小幅奖励，原来的进框、瞄准、自碰撞惩罚仍保留：

- `arm_drive_progress`：只在持球已就绪、仍接触球且无碰撞时累计。测量掌部相对机械臂基座朝目标的位移、肩肘前三个关节对掌部平移速度的贡献，以及球朝目标的速度。三者同时增加才有收益。
- `arm_drive_release`：确认有效且停车条件合格的释放后，只支付一次，以最后接触时的前送质量计算。先前送再停住或收回、最后仅靠手掌抛出，不会拿到同等释放奖励。

质量为 `clip(前送位移/0.12,0,1) × clip(肩肘前向速度/0.6,0,1) × clip(球前向速度/0.6,0,1)`。单位为米与米/秒。这些是归一化尺度，不是硬性轨迹或必须完成的距离。前三个关节贡献来自 MuJoCo 平移 Jacobian 与实测关节速度，底盘移动和单独转动手腕不能直接获得这一速度项。进度只奖励历史最佳质量的增长，最多 2 分；释放项最多另加 2 分，均明显低于进框奖励。所有旧配置默认此项为零。

## 探索性物理测试

记录在 `../analysis_artifacts/basket_grip_20261028/arm_probe.json`。使用同一 held 权重、新手型，等待一秒；测试 0.4/0.8/1.2 m/s 的肩肘前送请求和 0.12/0.24/0.36 秒时长，每组合三个种子，共 27 次，另有三次原策略对照。用阻尼 Jacobian 求解前三关节动作，前送期间保持手指目标，之后交回原策略。

部分动作组合增加射程，但大量组合更差。例如 0.8 m/s、0.24 秒组平均飞行距离约 0.80 m，进框 0/3；0.4 m/s、0.36 秒组进框 2/3。小样本只证明开手时序与前送必须协调，不能用来认定某个轨迹更优。因此正式训练不植入这些固定动作，保留策略选择权，让新增奖励鼓励协调。

新配置已完成双环境 CPU 短训练检查，采样、奖励计算、策略更新与保存跑通。尚未进行长时间训练，不能声称新策略已经学会更好的前送动作。正式评估应同时看安全进框率、自碰撞率、最后接触时的前送位移和速度，不能只看总奖励或动作幅度。

## 训练

```bash
cd /home/isee/Catch/catch_it_copy
python3 train_DCMM.py --config-name basket_hold_arm \
  test=False num_envs=32 vector_mode=async device_id=0 \
  viewer=false basket_log=false basket_self_collision_debug=false \
  basket_warm_start=/home/isee/Catch/catch_it_copy/outputs/BasketHold_03_06/2026-10-28/11:38:10/nn/throw_basket_catch_best_reward_-3.42.pth \
  train.ppo.max_agent_steps=2000000 output_name=BasketHoldArm
```

如果已经有新手型训练得到且实测更好的 held 权重，可将 `basket_warm_start` 换成它的完整路径。它是加载策略后重新训练价值网络的初始化入口，不是精确恢复优化器的断点续训。

新权重测试用同一 `--config-name basket_hold_arm` 和 `checkpoint_catching=新权重完整路径`，不需要姿态覆盖开关。旧权重不能直接以本配置测试来验证训练效果，因为本配置改变了奖励任务，元数据校验会拒绝；应训练后再评估。原配置继续可用。
