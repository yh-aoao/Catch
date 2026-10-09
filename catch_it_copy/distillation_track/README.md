# Track 在线蒸馏

从 `catch_it_copy` 目录运行 `python -m distillation_track.run`。这是独立的 Tracking 学生；与 `distillation` 的 Catch 学生使用不同的输入、输出和 checkpoint schema，不能互相续训，也尚未实现 Track 学生到 Catch 学生的权重迁移。

## 当前教师

| 任务 | best_model_track 文件 | 观测 / 动作 | 对应原程序 | 状态 |
|---|---|---|---|---|
| throw | throw_track_best_reward_54.78.pth | 18 / 8 | PPO ppo_dcmm；主环境 Tracking | 使用完整 base2 + arm6 接口 |
| roll | roll_track_best_reward_33.28.pth | 18 / 8 | PPO 645edc4；主环境内 RollTrackingEnv（07c176f） | 原推理、观测和动作转换校验通过 |
| bounce | bounce_track_best_58.18.pth | 18 / 8 | PPO 与环境均为 july22（58.18） | 原推理、观测和动作转换校验通过 |

配置为三个真实文件记录 SHA256。当前三个教师均为18维观测、8维动作。更换教师时必须同步审查配置和哈希，不能复用旧训练断点。旧的6维 `track.pth` 不再作为 throw 教师。默认检查和训练全部三个任务，也可用 `--tasks` 选择子集。

## 蒸馏逻辑

- 学生 MLP：29 → 256 → 256 → 128 → 8，隐藏层 ELU，输出 tanh。
- 输入：18 维 Tracking 状态 + 上一步实际执行的 8 维归一化动作 + 3 维任务标签。状态顺序与原 PPO_Track 一致：底座速度2、末端位置3/四元数4/速度3、物体位置3/速度3。没有手指观测，也没有新增桌子观测。
- 输出：底座2 + 机械臂6。动作缩放沿用原 Tracking 的底座1.5、机械臂0.025；送给环境的手指指令为12个零，手部状态由原 Tracking 环境控制。
- 每个任务建立独立环境池，`task='Tracking'`、`object_motion=任务名`；对应的冻结教师给学生访问到的状态标注动作。
- 默认在线 DAgger，beta=0，学生执行动作。不是继续 PPO，也不使用离线 BC 预训练。
- 教师保留各自冻结的18维 RunningMeanStd；学生先采集每环境256步校准状态统计量，然后冻结。动作历史和任务标签不参与标准化；回合结束后历史动作清零。
- 每任务默认2个环境，每轮每环境64步，32次梯度更新，每次每任务256样本，每任务回放容量100000。底座和机械臂分别计算组内均方误差，再等权平均；任务通过等量抽样平衡。
- 每25轮及最后一轮评估学生，默认每任务30回合；训练前先评估教师30回合。最佳模型按最差任务相对教师的成功率差距选择。

## 成功判定

roll、bounce 沿用各自原 Track 测试程序口径；throw 使用有效接触指标，修复当前原程序读取默认 false 的问题：

- roll：`roll_eval_success`，Tracking 的拦截判定，不是 Catch 的持续持球判定。
- bounce：july22 PPO_Track 使用 `truncated`；包含接触触发结束与超时，因此不能把100%解释为100%有效接触。
- throw：`track_contact_success = step_touch and not terminated`。主环境 Throw Tracking 没有更新默认 `success=False`，不能用该字段评估教师；仅超时不算成功，同步失败终止优先。原始 `success` 保留为辅助诊断，因此两者可能不同。

所有任务额外记录 `track_touch`、`track_contact_success = step_touch and not terminated`、原始 `success`（若有）、终止原因及超时标志。这些是诊断信息，不改写环境奖励或主指标。结果保存在 JSON 与逐回合 `.episodes.jsonl` 中。教师主指标为零时默认停止训练；确实要排查流程时才用 `--allow-zero-teacher-success`，该选项不会修正或提升成功率。

## 服务器运行

先检查接口，再测教师闭环：

```bash
python -m distillation_track.run check --tasks throw roll bounce
python -m distillation_track.run teacher-eval --tasks throw roll bounce --device cuda:0 --episodes 30 --output distillation_track/runs/trb_teacher
```

确认教师表现后，先运行5轮检查：

```bash
python -m distillation_track.run train --tasks throw roll bounce --device cuda:0 --iterations 5 --num-envs 2 --output distillation_track/runs/trb_smoke
```

代码、环境、教师和其他参数不变时，将同一实验续训到200轮：

```bash
python -m distillation_track.run train --tasks throw roll bounce --device cuda:0 --iterations 200 --num-envs 2 --output distillation_track/runs/trb_smoke --resume distillation_track/runs/trb_smoke/resume_last.pth
```

独立评估最佳学生（无需教师文件）：

```bash
python -m distillation_track.run eval --checkpoint distillation_track/runs/trb_smoke/student_best.pth --device cuda:0 --episodes 100 --output distillation_track/runs/trb_eval
```

可在 `teacher-eval` / `eval` 添加 `--viewer` 观察动作。新实验使用新的 `--output`；已有学生的目录不允许无断点覆盖。输出包括 `resolved_config.json`、`teacher_check.json`、`teacher_baseline.json`、`metrics.jsonl`、定期评估结果、`student_best.pth`、`student_last.pth` 和 `resume_last.pth`。续训恢复优化器、回放和随机数状态，但模拟器开始新回合；只允许改变总迭代数。评估读取 checkpoint 内嵌配置。

## 验证边界

```bash
python -m unittest discover -s distillation_track/tests -v
```

测试覆盖真实教师的严格恢复与推理一致性、动作缩放、零手指指令、掩码损失、跨回合动作重置、真实教师加模拟环境的训练/续训/学生独立评估，以及独立进程的环境错误与终止信息传播。模拟环境测试不代表 MuJoCo 闭环成功率。真实闭环与CUDA训练需在项目可运行的服务器环境验证。


## 无显示服务器的 bounce 渲染

Track 学生只使用18维状态，不读取相机图像。配置中的 `tasks.bounce.env_kwargs.render_mode: null` 让 July22 环境跳过离屏渲染器创建和相机渲染，避免无可用 OpenGL 上下文时出现 `gladLoadGL error`。显式选择 RGB/深度模式时仍保留原渲染路径；`--viewer` 仍需要服务器具备图形显示条件。

更新时同时同步配置和 `gym_dcmm/envs/DcmmVecEnv_bounce_july22.py`。先运行一次三任务教师评估，再启动5000轮训练。若此前在初始化阶段失败、输出目录没有学生 checkpoint，可重用该目录；已开始训练的旧断点因配置/源码变化不能直接续训，应使用新目录。此修复不改变动力学、奖励、动作或成功判定。
