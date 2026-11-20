# 网球末端位姿增量训练版本

本版本延续网球单球两阶段训练。机械臂输出三个末端位置增量和三个姿态增量，经现有实测状态 IK 转为关节目标。位置每轴最大 0.025 米，姿态每轴最大 0.025 弧度；姿态沿用现有 Z、X、Y 欧拉增量顺序。手指仍为 12 维动作；第二阶段增加 2 维底盘动作。

第一阶段 `basket_tennis_range_ee` 从零训练固定底座抛球，目标框每回合采样后保持静止，前后范围沿用原网球 range 配置。第二阶段 `basket_mobile_tennis_free_stop_ee` 加载新第一阶段权重，模型在允许的停车区域内学习停靠位置，底盘和抛球策略都继续训练，并保留固定底座回合。奖励函数、持球姿势和停车规则沿用上一版。

旧关节速度权重不能用于此模式，加载时会检查控制类型。旧 joint_velocity 配置保留。没有启动正式训练。

## 训练命令

第一阶段：

```bash
cd ~/Catch/catch_it_copy
python3 train_DCMM.py --config-name basket_tennis_range_ee \
  test=False num_envs=32 vector_mode=async device_id=0 \
  basket_mixed_source='' checkpoint_catching='' checkpoint_tracking='' basket_warm_start='' \
  viewer=false basket_log=false basket_self_collision_debug=false \
  train.ppo.max_agent_steps=1000000000000 \
  output_name=BasketTennisRangeEEScratch
```

第一阶段训练好后，将下方占位路径换成新末端增量网球权重的完整路径，再启动第二阶段：

```bash
python3 train_DCMM.py --config-name basket_mobile_tennis_free_stop_ee \
  test=False num_envs=32 vector_mode=async device_id=0 \
  basket_throw_checkpoint='/第一阶段末端增量网球模型的完整路径.pth' \
  checkpoint_catching='' basket_mixed_source='' basket_warm_start='' \
  viewer=false basket_log=false basket_self_collision_debug=false \
  train.ppo.max_agent_steps=1000000000000 \
  output_name=BasketTennisFreeStopEE
```

## 同步到服务器

需要同步以下运行时代码及两个新配置，不能只同步 YAML：

- `train_DCMM.py`
- `gym_dcmm/envs/DcmmVecEnv.py`
- `gym_dcmm/envs/basket_mixed.py`
- `gym_dcmm/algs/ppo_dcmm/ppo_dcmm_basket_mixed.py`
- `gym_dcmm/algs/ppo_dcmm/ppo_dcmm_basket_mobile_joint.py`
- `gym_dcmm/algs/ppo_dcmm/ppo_dcmm_catch_one_stage.py`
- `gym_dcmm/algs/ppo_dcmm/basket_mobile_joint.py`
- `gym_dcmm/algs/ppo_dcmm/basket_mixed_policy.py`
- `gym_dcmm/utils/basket_throw_policy.py`
- `configs/basket_tennis_range_ee.yaml`
- `configs/basket_mobile_tennis_free_stop_ee.yaml`

## 验证与回退

55 项测试通过，包括实际 IK 的位置和姿态增量、两阶段导入、保存恢复、控制类型检查和原关节速度路径回归；两个阶段的异步环境入口推理检查通过。上述检查不代表训练成功率，仍需实际训练评估。

修改前快照：`backups/before_tennis_ee_delta_20261120_164846/`。在项目目录运行以下命令恢复本次修改的九个既有运行时文件（新配置会保留）：

```bash
bash backups/before_tennis_ee_delta_20261120_164846/rollback.sh
```
