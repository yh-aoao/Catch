# 六球 Bounce / 七目标 Roll：物理参数泛化 v1

独立可选入口：`scripts/train_all_targets_dr_v1.sh`。旧训练命令默认关闭随机化，
原有目标目录、奖励、动作和观测定义不变。旧源码备份在
`backups/before_domain_randomization_20261105/`。

## 随机化设计

仿照 Throw，**每个并行环境创建时采样一次**，该环境后续各回合固定这套参数。
目标仍按环境均分，每次 reset 随机发射。不是每回合重新采样物理参数。

|参数|Roll|Bounce|
|---|---|---|
|尺寸统一缩放|0.95～1.05 倍|半径 0.95～1.05 倍|
|质量|0.90～1.10 倍|0.90～1.10 倍|
|物体/运动表面滑动摩擦|0.85～1.15 倍|0.85～1.15 倍|
|滚动阻力长度|0.80～1.20 倍|保持基准|
|物体/地面接触阻尼比|保持基准|0.90～1.10 倍|

以上独立均匀采样。质量和尺寸独立变化，相当于也覆盖密度差异。
惯量同步为 `I_new = I_base × mass_scale × size_scale²`；外观和碰撞形状同步缩放。
水果保持凸包形状，罐保持侧放圆柱，仅统一缩放，不改变长宽比例。
不增加自由关节阻尼，不改变重力，不加持续外力。
未随机化机器人手部表面摩擦、时间常数、接触求解器与惯量结构系数。

Bounce 扰动的是 **MuJoCo 接触阻尼比**，不是直接随机 COR。
目录中的 `normal_cor_target` 仍是基准标记；不能把它当成采样后的实测弹性。
发射器保留原有基准能量设计和随机轨迹，实际反弹随阻尼变化。
因此并不保证每个扰动样本都具有完全相同的可达性或相同反弹高度。

默认 Roll 28 环境、Bounce 24 环境，各目标只有 4 套固定参数；若资源足够，可用
56/48 环境增加到每目标 8 套。换 seed 也会换物理样本。
`object_benchmark_provenance.json` 保存每个环境的目标、尺度与源码/目录摘要。
只复制模型不够完整复现，请保留训练目录中的 resolved_config.yaml 与 provenance。

## 第一阶段：从头训练 Track

```bash
conda activate dcmm
cd /home/isee/Catch/catch_it_copy
bash scripts/train_all_targets_dr_v1.sh roll track device_id=0
bash scripts/train_all_targets_dr_v1.sh bounce track device_id=0
```

输出：

- `outputs/roll_seven_object_tracking_dr_v1_seed123/`
- `outputs/bounce_six_ball_tracking_dr_v1_seed123/`

默认沿用此前 horizon=64、均分目标、最大步数 10 亿；minibatch 自动按环境数调整。
可添加 `num_envs=56`（Roll）或 `num_envs=48`（Bounce）。

也可以在 Track 命令后指定旧 Track checkpoint 做权重初始化；这与从头训练不同，
建议单独命名 output_name，避免混淆实验。

## 第二阶段：从新 Track 初始化 Catch

```bash
bash scripts/train_all_targets_dr_v1.sh roll catch device_id=0 \
  checkpoint_tracking=/实际目录/roll_track_best_reward_XX.pth
bash scripts/train_all_targets_dr_v1.sh bounce catch device_id=0 \
  checkpoint_tracking=/实际目录/bounce_track_best_reward_XX.pth
```

Catch 必须显式提供 Track 文件，防止误用旧版默认模型。
输出为 `roll_seven_object_catching_dr_v1_seed123` 和
`bounce_six_ball_catching_dr_v1_seed123`。Track/Catch 联合训练，不冻结 Track。
Roll 沿用现有 legacy 抓取奖励；Bounce 沿用 controlled_hold_v1 和现有手指奖励。

## 对照评估

训练完成后分别评估：旧固定参数、训练参数采样、新 seed 参数采样。
测试必须显式选择同一个完整目标目录及是否打开 DR；原有 viewer 命令不会自动开启 DR。
不能仅凭训练日志宣称物理泛化已提升。当前新增验证只证明场景与训练入口可运行，
不代表已经训练出新策略或验证真实物体成功率。
