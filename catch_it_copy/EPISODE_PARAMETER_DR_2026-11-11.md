# 每回合物理参数随机化：Roll 七目标 / Bounce 六球

新开关：`benchmark_options.domain_randomization=physical_dr_episode_v1`。
旧的 `physical_dr_v1` 仍然是创建环境时采样，旧脚本不会自动切换。

每次手动 reset 和并行环境的自动回合重置都会重新采样。单个回合内参数不变，物体类别仍固定在环境槽位上，保持七类/六类平衡。随机范围不变：尺寸±5%、质量±10%、滑动摩擦±15%、Roll滚动摩擦长度±20%、Bounce接触阻尼比±10%。阻尼比不是恢复系数。

同步更新质量、惯量（质量倍率×尺寸倍率平方）、碰撞尺寸、网格顶点/包围树、可视网格、表面接触参数；Roll重新计算桌面初始高度与无滑动角速度，Bounce保留原发射分布。整个回合不额外推球。没有修改奖励、成功条件、物体分配和默认发射范围。

代码：`ycb_objects/episode_randomization.py`、`gym_dcmm/envs/object_benchmark.py`。
统一入口：`scripts/train_all_targets_episode_dr_v1.sh`。

## 推荐：在现有完整 Catch 模型上继续训练

```bash
conda activate dcmm
cd /home/isee/Catch/catch_it_copy

bash scripts/train_all_targets_episode_dr_v1.sh roll finetune \
  checkpoint_catching=/home/isee/Catch/11-11/roll_catch_best_reward_231.61.pth \
  device_id=0

bash scripts/train_all_targets_episode_dr_v1.sh bounce finetune \
  checkpoint_catching=/home/isee/Catch/11-11/bounce_catch_best_reward_4.84.pth \
  device_id=0
```

两条命令分别运行；只有一张GPU时请分开运行。默认Roll28环境、Bounce24环境，每类4个槽位，槽位物理参数每回合更新；默认学习率1e-4，Track/Catch联合更新，不冻结Track。加载完整模型和归一化统计；优化器、计数、best记录重新开始，属于新实验而非断点恢复。可以用 `checkpoint_catching=/其他路径/model.pth` 替换输入路径。也可以使用兼容的旧三球Catch，例如Bounce13.68，但新增物体仍需适应，不能保证表现提升。

输出：
- `outputs/roll_seven_object_catching_episode_dr_from_catch_v1_seed123/`
- `outputs/bounce_six_ball_catching_episode_dr_from_catch_v1_seed123/`

## 可选：重新执行 Track → Catch 两阶段

```bash
bash scripts/train_all_targets_episode_dr_v1.sh roll track device_id=0
bash scripts/train_all_targets_episode_dr_v1.sh bounce track device_id=0
```

Track完成后，填写真实的Track checkpoint文件路径：

```bash
bash scripts/train_all_targets_episode_dr_v1.sh roll catch \
  checkpoint_tracking=/完整路径/roll_track_best_reward_XX.pth device_id=0

bash scripts/train_all_targets_episode_dr_v1.sh bounce catch \
  checkpoint_tracking=/完整路径/bounce_track_best_reward_XX.pth device_id=0
```

Track从头训练，Catch加载Track并新建Catch分支。输出前缀分别为 `${motion}_seven_object` / `bounce_six_ball` 的 `tracking_episode_dr_v1_seed123` 和 `catching_episode_dr_v1_seed123`（Roll前缀为roll_seven_object）。

最大步数默认1,000,000,000，只是宽松上限。可覆盖 `train.ppo.max_agent_steps=...`、`seed=...`、`output_name=...`、`num_envs=...` 等。改变seed时建议同步改output_name，默认输出名中的seed123不会自动替换。

## 每回合随机参数的独立评估

评估清单新模式：`"physics":"dr_episode"`。`"dr"`仍使用旧的每环境采样，`"fixed"`仍使用固定参数。

```bash
python scripts/evaluate_multitarget_catch.py \
  --models-file scripts/eval_episode_dr_1111.json \
  --episodes 200 --instances 20 --jobs 2 \
  --output evaluation_results/episode_dr_1111_200
```

清单默认评估现有231.61/4.84模型；训练后请把checkpoint路径改成新模型，并使用新的输出目录。每个目标200回合，20个槽位每槽10回合，每回合物理参数均重新采样。`episodes.jsonl`保存每回合真实参数，summary中的parameter_samples在dr_episode模式下对应实际各回合样本。不要使用旧Bounce预计算viewer替代本版本物理采样评估。

## 备份与验证

修改前文件：`/home/isee/Catch/backups/before_episode_dr_20261111/`。
旧物理模式和旧训练入口保留；删除新代码不是切回旧模式的必要步骤。
验证记录：`/home/isee/Catch/analysis_artifacts/episode_dr_20261111/`。
没有启动正式长训练。短程PPO/评估仅验证实现，不代表成功率已经提升。
