# Roll 三球双阶段重训

脚本：`scripts/train_roll_three_ball_v1.sh`。
保持既有 Tracking → Catching_TwoStage 流程，不改 PPO、网络结构、动作接口或抓取判据。
启用已验证的 roll_physics=physical_v1 与 adaptive_wait_v1 等待奖励。
旧模型和旧物理配置保留，新训练输出使用独立目录和日期/时间。

## 配置

- 每回合按原始参考球 60%、网球 20%、棒球 20% 采样，不是固定数量配额。
- 三球分别使用 balls.roll_table_priors_v1.json 的质量、尺寸、惯量、滑动及滚动阻力。
- 当前是三组固定动力学先验之间的混合；不额外随机扰动材质参数、质量或半径。
- 初始速度统一在 0.8–1.3 m/s 采样，方向及初始位置沿用 Roll 环境分布。
- 默认 16 个环境、GPU 0、seed=123，每个阶段默认上限 2500 万 agent steps。
- 两阶段物理和采样配置一致；默认无 Viewer，状态教师训练不需要离屏 OpenGL。
- 材质是文献/工程近似；混合训练沿用同一个解析球外观，是状态教师动力学训练，
  不是三种 YCB 纹理外观的视觉学生训练。

## 第一阶段：Tracking 从头训练

```bash
conda activate dcmm
cd /home/isee/Catch/catch_it_copy
bash scripts/train_roll_three_ball_v1.sh
```

输出：`outputs/roll_three_ball_tracking_v1_seed123/日期/时间/nn/`。
选择实际验证效果较好的 `roll_track_best_reward_*.pth`，或使用 `roll_track_last.pth`。
不要把文档里的占位路径直接当作真实 checkpoint。

## 第二阶段：加载新 Track，训练 Catch

第一阶段训练完成后，把路径替换为其实际输出：

```bash
bash scripts/train_roll_three_ball_v1.sh \
  task=Catching_TwoStage \
  checkpoint_tracking=/actual/path/nn/roll_track_last.pth
```

脚本会自动使用 `output_name=roll_three_ball_catching_v1_seed123`。
第二阶段导入 Tracking actor 和输入归一化，抓取分支重新初始化，按既有 TwoStage PPO 训练。
脚本不会自动选择 Track checkpoint，也不会在 Track 未训练好时自动开始 Catch。
未提供权重、路径不存在或误选 OneStage 会提前报错。
允许显式传兼容的 checkpoint_catching 接着训练，但这是已有模型初始化，不是从头双阶段重训。

GPU/环境数覆盖放在命令后，两阶段都应使用相同物理相关覆盖：

```bash
bash scripts/train_roll_three_ball_v1.sh device_id=1 num_envs=16
```

每次运行仍按日期/时间新建输出目录；训练额外保存 resolved_config.yaml、
roll_physics_provenance.json、roll_reward_provenance.json 与奖励分项日志。
正式训练尚未由助手启动。

## 验证范围

脚本已通过 shell 语法与误用检查；CPU 双异步环境的短 Tracking 训练保存了新 checkpoint。
随后 Catching_TwoStage 成功加载该新 Track actor 与归一化，并完成 2 次短 PPO 更新。
链路日志：`/tmp/roll_three_ball_tracking_script.log`、`/tmp/roll_three_ball_catching_script.log`。
训练链路短检查不能证明收敛或泛化，必须分别评估三种球，避免混合均值掩盖弱项。
之前物理版本的 21 项核心回归、离桌及滚动阻力对照见 ROLL_PHYSICAL_V1_2026-10-27.md。

## 测试最终 Catch 模型

```bash
python train_DCMM.py test=True task=Catching_TwoStage num_envs=1 \
  object_motion=roll roll_physics=physical_v1 \
  roll_reward_version=adaptive_wait_v1 roll_wait_guard=false \
  roll_material_file=/home/isee/Catch/catch_it_copy/ycb_objects/calibration_data/balls.roll_table_priors_v1.json \
  'roll_speed=[0.8,1.3]' test_object=ycb:tennis_ball ycb_collision=sphere \
  object_eval=false viewer=true checkpoint_catching=/actual/path/to/new_catch.pth
```

棒球改 ycb:baseball；原始参考球改 test_object=default 并去掉显式 ycb_collision。
固定测试不传混合采样参数或权重。
