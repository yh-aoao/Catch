# 七目标 Roll / 六球 Bounce：第二阶段 Catch

```bash
conda activate dcmm
cd /home/isee/Catch/catch_it_copy
bash scripts/train_all_targets_catch_v1.sh roll device_id=0
bash scripts/train_all_targets_catch_v1.sh bounce device_id=0
```

两个命令分别在终端运行；同时训练时按实际显存分配 GPU。

默认加载本地 `outputs/*tracking_v1_seed123/2026-11-04/01:33:28/nn/` 中
Roll `roll_track_best_reward_24.83.pth`、Bounce `bounce_track_best_reward_19.08.pth`。
通过 `checkpoint_tracking=/绝对路径/model.pth` 可替换。

这是从 Track 开始的新 Catch 训练，不是恢复旧 Catch。加载 Track actor、输出头、
动作标准差与观测归一化统计，Catch 分支和联合 critic 重新初始化。
Track 与 Catch 联合优化；Bounce 显式设置 freeze_track_steps=0、track_lr_scale=1、
freeze_track_normalization=false，沿用此前较好模型的训练设置。

目标、物理、发射与第一阶段保持一致，完整目录为
`ycb_objects/calibration_data/object_benchmark_all_targets_v1.json`。
Roll 28 个环境，每种目标 4 个；Bounce 24 个环境，每种球 4 个。
每个环境固定一种目标，每次 reset 随机发射；相同数量的 rollout 交互，
不保证每种物体完成的回合数量相同。

Roll 沿用现有 legacy 抓取奖励，不启用硬等待限制；Bounce 沿用
controlled_hold_v1 及已有手指奖励（backfold=.15、prepare_sync=.05、
contact=1、thumb_opposition=.5），不是旧版距离成功奖励。本入口不修改奖励源码。

默认 seed=123、horizon=64、minibatch=环境数×horizon/2、最大步数 10 亿。
这是停止上限，不要求必须训练到上限。输出分别为：

- `outputs/roll_seven_object_catching_v1_seed123/`
- `outputs/bounce_six_ball_catching_v1_seed123/`

降低资源占用且保持均分：

```bash
bash scripts/train_all_targets_catch_v1.sh roll device_id=0 num_envs=14
bash scripts/train_all_targets_catch_v1.sh bounce device_id=0 num_envs=12
```

本脚本只允许从 Track 初始化；恢复 Catch 请使用基础入口
`train_object_benchmark.sh`，显式指定完整目标目录和 Catch checkpoint。
Roll 非球形物体仍沿用现有抓取控制与奖励，Track 拦截成功不保证 Catch 能抓稳。
