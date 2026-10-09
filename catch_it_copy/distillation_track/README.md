# Track 在线蒸馏

从 `catch_it_copy` 目录运行 `python -m distillation_track.run`。这是独立的 Tracking 学生；与 `distillation` 的 Catch 学生使用不同的输入、输出和 checkpoint schema，不能互相续训，也尚未实现 Track 学生到 Catch 学生的权重迁移。

## 当前教师与阻塞项

| 任务 | best_model_track 文件 | 观测 / 动作 | 对应原程序 | 状态 |
|---|---|---|---|---|
| throw | track.pth | 18 / 6 | 待恢复原控制器 | 权重可严格加载，动作接口尚未确认，禁止执行 |
| roll | roll_track_best_reward_33.28.pth | 18 / 8 | PPO 645edc4；主环境内 RollTrackingEnv（07c176f） | 原推理、观测和动作转换校验通过 |
| bounce | bounce_track_best_58.18.pth | 18 / 8 | PPO 与环境均为 july22（58.18） | 原推理、观测和动作转换校验通过 |

配置为三个真实文件记录 SHA256。更换教师时必须同步审查配置和哈希，不能复用旧训练断点。throw 的 6 维输出不能从权重形状推断出每个控制轴；当前 `action_indices: null` 会明确阻止执行，不能简单改成 `[0,1,2,3,4,5]`。需要原模型可运行的测试命令和控制器，或当前环境训练出的 8 维 throw Track checkpoint。默认不传 `--tasks` 会检查全部三个任务并在 throw 接口处停止；现阶段可显式选择 `--tasks roll bounce`。

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

沿用当前各自原 Track 测试程序的口径，而非 Catch 的统一指标：

- roll：`roll_eval_success`，Tracking 的拦截判定，不是 Catch 的持续持球判定。
- bounce：july22 PPO_Track 使用 `truncated`；包含接触触发结束与超时，因此不能把100%解释为100%有效接触。
- throw：配置暂沿用当前 PPO_Track 的 `success`，但其6维旧教师原环境尚未核实；接入时还需核对原终止口径。

所有任务额外记录 `track_touch`、`track_contact_success = step_touch and not terminated`、原始 `success`（若有）、终止原因及超时标志。这些是诊断信息，不改写环境奖励或主指标。结果保存在 JSON 与逐回合 `.episodes.jsonl` 中。教师主指标为零时默认停止训练；确实要排查流程时才用 `--allow-zero-teacher-success`，该选项不会修正或提升成功率。

## 服务器运行

先检查接口，再测教师闭环：

```bash
python -m distillation_track.run check --tasks roll bounce
python -m distillation_track.run teacher-eval --tasks roll bounce --device cuda:0 --episodes 30 --output distillation_track/runs/rb_teacher
```

确认教师表现后，先运行5轮检查：

```bash
python -m distillation_track.run train --tasks roll bounce --device cuda:0 --iterations 5 --num-envs 2 --output distillation_track/runs/rb_smoke
```

代码、环境、教师和其他参数不变时，将同一实验续训到200轮：

```bash
python -m distillation_track.run train --tasks roll bounce --device cuda:0 --iterations 200 --num-envs 2 --output distillation_track/runs/rb_smoke --resume distillation_track/runs/rb_smoke/resume_last.pth
```

独立评估最佳学生（无需教师文件）：

```bash
python -m distillation_track.run eval --checkpoint distillation_track/runs/rb_smoke/student_best.pth --device cuda:0 --episodes 100 --output distillation_track/runs/rb_eval
```

可在 `teacher-eval` / `eval` 添加 `--viewer` 观察动作。新实验使用新的 `--output`；已有学生的目录不允许无断点覆盖。输出包括 `resolved_config.json`、`teacher_check.json`、`teacher_baseline.json`、`metrics.jsonl`、定期评估结果、`student_best.pth`、`student_last.pth` 和 `resume_last.pth`。续训恢复优化器、回放和随机数状态，但模拟器开始新回合；只允许改变总迭代数。评估读取 checkpoint 内嵌配置。

## 验证边界

```bash
python -m unittest discover -s distillation_track/tests -v
```

测试覆盖真实教师的严格恢复与推理一致性、动作缩放、零手指指令、掩码损失、跨回合动作重置、真实教师加模拟环境的训练/续训/学生独立评估，以及独立进程的环境错误与终止信息传播。模拟环境测试不代表 MuJoCo 闭环成功率。真实闭环与CUDA训练需在项目可运行的服务器环境验证。
