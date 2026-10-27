# Bounce 三球双阶段训练 v1

就绪配置：physical_v2，保持当前 Bounce 奖励和 PPO，三种解析球动力学先验。
按 episode 采样原始球 60%、网球 20%、棒球 20%；不是精确回合配额或帧数比例。
每类球质量再扰动 ±10%、半径 ±5%，惯量按各自结构近似随 m*r² 变化。
只为棒球使用 baseball_intercept_v1 发射配置；其他球保持 physical_v2 默认发射条件。

独立材质文件：`ycb_objects/calibration_data/balls.three_ball_training_v1.json`。
原始球保留 physical_v2 的仿真材质先验；网球和棒球继承文献近似配置，来源与不确定性
保存在文件。三球原始质量/半径及惯量模型可在 provenance 中追溯。
这是状态教师的多物理参数训练，训练视觉沿用同一个解析球；不是 YCB RGB 外观混合训练。

## 启动

```bash
conda activate dcmm
cd /home/isee/Catch/catch_it_copy
bash scripts/train_bounce_three_ball_v1.sh
```

脚本默认：第一阶段 Tracking，从头训练，无 Track/Catch checkpoint；16 个环境，GPU 0，
seed=123，PPO 默认 2500 万 agent steps。输出到独立 `outputs/bounce_three_ball_tracking_v1_seed123/日期/时间/`。
同名运行仍按日期/时间分目录，不覆盖旧模型。

Tracking 训练完成后，选择其 `nn/bounce_track_best_reward_*.pth` 或 `bounce_track_last.pth`，
再启动第二阶段（将示例替换为实际绝对路径）：

```bash
bash scripts/train_bounce_three_ball_v1.sh \
  task=Catching_TwoStage \
  checkpoint_tracking=/actual/path/nn/bounce_track_last.pth \
  output_name=bounce_three_ball_catching_v1_seed123
```

第二阶段加载 Track actor 和输入归一化，初始化抓取分支，再按既有 TwoStage PPO 训练。
两个阶段都使用相同三球比例、材料配置和棒球专用发射设置。
脚本不会在 Track 训练未完成时自动选择 checkpoint、接着开始 Catch。
旧 OneStage 训练输出保留，它与 TwoStage 网络结构不同，不作为 TwoStage 的续训 checkpoint。

调整环境数、GPU 或其他 Hydra 配置，可将覆盖参数放在脚本后：

```bash
bash scripts/train_bounce_three_ball_v1.sh device_id=1 num_envs=16
```

CPU 用 `device_id=-1`，但本次只在 CPU 做过短链路验证，未测正式训练吞吐。
本地 NVIDIA-SMI 查询未成功，不能宣称本机 GPU 训练已验证；GPU 训练机需正常可用的 CUDA。

## 验证与版本

9 项相关测试通过，包括权重校验、实际环境三球采样、棒球选择性发射、
不同材料逐回合切换、质量/尺寸/惯量关联、重置姿态及控制周期保持。
先前 OneStage 脚本只验证了单阶段链路，不能代表 Track→Catch 已验证；当前默认已纠正为 Tracking。
纠正后已在 CPU 双异步环境完成 Tracking 的 2 次短更新、保存 checkpoint，
随后由 Catching_TwoStage 成功加载该 Track 权重和归一化，再完成 2 次短更新。
日志分别为 `/tmp/bounce_three_ball_tracking_smoke.log` 和
`/tmp/bounce_three_ball_two_stage_smoke.log`。这仍只是链路验证，不代表模型已学会抓取。
resolved_config.yaml 和 bounce_physics_provenance.json 保存参数、材质快照和源码 SHA256。
默认配置没有自动启用本训练；不传新脚本参数时旧测试方式保持原样。
本轮未启动长期训练，也未改动旧 checkpoint。

## 测试新模型

按本双阶段流程得到的最终 Catch 模型使用 `task=Catching_TwoStage` 测试。
仅先前 OneStage 训练得到的模型才使用 Catching_OneStage，两类模型不要混用。
分别固定 original/tennis_ball/baseball 做测试，而不是只看训练混合平均分。

```bash
python train_DCMM.py test=True task=Catching_TwoStage num_envs=1 \
  object_motion=bounce bounce_physics=physical_v2 bounce_launch=baseball_intercept_v1 \
  bounce_material_file=/home/isee/Catch/catch_it_copy/ycb_objects/calibration_data/balls.three_ball_training_v1.json \
  test_object=ycb:baseball ycb_collision=sphere object_eval=false viewer=true \
  checkpoint_catching=/absolute/path/to/new_checkpoint.pth
```

网球改 `test_object=ycb:tennis_ball`；原始球用 `test_object=default` 并去掉显式 ycb_collision。
固定评估不传 bounce_ball_weights；默认 bounce_ball_mode=fixed。
