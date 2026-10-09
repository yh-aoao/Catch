# Throw / Roll / Bounce：共享 MLP 在线 DAgger

实现日期：2026-10-02。第一版使用一个学生 MLP `[256,256,128]`，直接在线 DAgger，默认 `beta_start=0`，不包含离线 BC 预热、PPO 微调或 GRU。只训练学生，不更新教师和原环境。

**2026-10-09 当前状态：默认 throw 教师已切换为服务器上的 throw_catch_best_reward_497.09.pth，配置要求 8+12=20 维动作，使用完整的一一映射。新权重本地无副本，实际形状、推理一致性及闭环表现需在服务器验证；不再使用旧 18 维 catch_two_stage.pth。roll/bounce 的成功指标校准问题仍保留。**


## 2026-10-08：评估接口修复与 roll 指标排查

bounce 实际路由到 BounceEnv。其 _get_info() 缺少默认 success，只有部分 step 分支写入该键。现在已在该具体类中补齐 success=False，明确成功分支仍覆盖为 True。奖励、控制和终止条件不变，未将超时当接球成功。同步服务器时，必须同时更新 gym_dcmm/envs/DcmmVecEnv.py 和 distillation 目录（含新增 metrics.py）。

roll_eval_success 是额外严格指标：手部接触、球离开桌面/地面、位于手部区域、相对速度不超过 0.15 m/s，并持续 0.30 s（短接触间隙容忍 0.04 s）。它与原环境成功分支不同。当前保留定义，同时记录 success、roll_legacy_success、roll_eval_success、roll_final_hold、roll_eval.max_duration、reset_counts 和 roll_eval_unmet，先定位统计偏差再校准。

评估在第 1、每 10 个和最终回合显示进度；每回合更新 JSON，并追加同名 .episodes.jsonl 终止诊断。后一个任务异常仍保留前一个任务结果。complete、requested_episodes、episodes 用于区分完整与部分结果。其他缺失字段仍报错，不静默把未知当失败。

先运行 10 回合诊断：

    python -m distillation.run teacher-eval --tasks roll bounce --device cuda:2 --episodes 10 --output distillation/runs/rb_teacher_diagnostic

需要先验证 5 轮流程时，显式允许零基准继续：

    python -m distillation.run train --tasks roll bounce --device cuda:2 --iterations 5 --num-envs 2 --allow-zero-teacher-success --output distillation/runs/rb_smoke_metrics_fix

--allow-zero-teacher-success 只解除零基准阻断，仍评估教师，不改写成功数、不跳过缺失字段错误，并将选项保存到配置。成功指标未校准时，student_best.pth 的成功率排序只能视为暂定，不能用于证明效果达标。正常非零基准无需该选项。

本次验证：11 项 CPU/模拟环境回归测试通过，包含实际 BounceEnv 方法的默认字段、原成功赋值保留、后续任务异常仍保留已完成结果、零指标显式试训。本机未完成真实 MuJoCo 教师效果验证。

## 设计和范围

- 教师：严格恢复完整 TwoStage Catch checkpoint 和各自的 18D/12D normalizer，调用原 `ActorCritic.act_inference`；不实例化 PPO，不加载旧优化器。缺失键、形状不匹配、hash 不符均报错。
- 学生输入 53D：原始状态 30D＋上一实际发送的归一化命令 20D＋固定任务 one-hot 3D。状态顺序与原 `obs2tensor` 对照检查；不改变原坐标系。
- 单个共享动作头输出 20D：base2、arm6、hand12，tanh 有界。每任务保留原动作缩放和控制器；这是任务条件化单网络，不是统一底层控制，也不自主识别任务。
- 每个仿真步：当前学生状态 → 对应教师标注 → 学生执行 → 存入该任务缓存。任务批次等量，底盘/机械臂/手三组 masked MSE 等权。
- 开始先在线采集 `calibration_steps=256` 步/环境，用均衡的实际访问状态拟合学生状态 normalizer 并冻结。校准阶段默认仍由学生执行，无监督更新；不是离线 BC。上一动作和 one-hot 不做统计归一化。
- 每任务独立 ring buffer，默认 10 万条；默认每任务 2 个 spawn 进程、每轮采样 64 步并更新 32 次、每任务 minibatch 256。无需满足 PPO 的 minibatch 乘积限制。
- 回合结束先保存终止 info，再 reset；上一命令清零，避免跨回合污染。工作进程异常向主进程传播，不吞掉异常。
- 原环境 `reset()` 不接收 seed，因此每个独立进程在每回合 reset 前设置 NumPy/Python seed。评估固定同一套回合 seed；这不保证所有仿真运算完全确定性。

论文方法来源：Robot Parkour Learning §3.2，https://robot-parkour.github.io/resources/Robot_Parkour_Learning.pdf 。这里只借鉴多教师在线 DAgger；MLP、显式标签、MSE 和本配置参数是项目改编，不是论文视觉 CNN＋GRU／BCE 配置。

## 先检查教师

以下命令均从 `catch_it_copy` 运行，在原项目可用的 Python 环境中执行。CPU 检查无需 MuJoCo，可使用 `python` 或 `python3`。

```bash
python -m distillation.run check --device cpu --output distillation/runs/interface_check
```

在服务器执行后，应检查三个教师均为 `native_action_dim=20`、`interface_ready=true`，且观测、动作和推理对照通过。报告在 `teacher_check.json`。缺少新文件或动作维度不符会明确停止，不会回退到旧 throw 权重。接口检查通过仍不等于接球效果复现。

## 在训练机器上先跑 roll/bounce

先复现两个教师的真实闭环（100 回合/任务）：

```bash
python -m distillation.run teacher-eval --tasks roll bounce --device cuda:0 --episodes 100 --output distillation/runs/rb_teacher
```

需要查看动作时使用单任务：

```bash
python -m distillation.run teacher-eval --tasks bounce --device cuda:0 --episodes 10 --viewer --output distillation/runs/bounce_view
```

确认教师动作与之前正常测试一致，再试跑 5 轮训练，验证采样、更新和保存：

```bash
python -m distillation.run train --tasks roll bounce --device cuda:0 --iterations 5 --num-envs 2 --output distillation/runs/rb_smoke
```

正式试训用新的输出目录：

```bash
python -m distillation.run train --tasks roll bounce --device cuda:0 --iterations 1000 --num-envs 2 --output distillation/runs/rb_mlp_seed123
```

每次新训练会先运行默认 30 回合/任务教师基准；某教师零成功会停止并要求检查基准。1000 轮、2 环境、64 步约为每任务 12.8 万在线 transitions，另加校准，不代表已足够收敛。按学习曲线延长预算。环境进程数是任务数乘 `--num-envs`，评估时另开 1 个进程；不要直接把每任务环境数都设成全部 CPU 数。

断点续训（配置一致，仅可延长 iterations）：

```bash
python -m distillation.run train --tasks roll bounce --device cuda:0 --iterations 2000 --num-envs 2 --output distillation/runs/rb_mlp_seed123 --resume distillation/runs/rb_mlp_seed123/resume_last.pth
```

恢复学生、优化器、缓存与主进程随机状态；仿真从新回合开始，不是逐比特续跑。源文件/资产 hash 改变时拒绝恢复，防止混用环境版本。

独立评估学生（不读取或构造教师模型；也不需要教师权重文件）：

```bash
python -m distillation.run eval --checkpoint distillation/runs/rb_mlp_seed123/student_best.pth --device cuda:0 --episodes 300 --output distillation/runs/rb_final_eval
```

## 三任务训练：接入新的 throw 教师（2026-10-09）

配置为 `configs/three_tasks.json`。throw 路径使用项目相对路径 `best_model/throw_catch_best_reward_497.09.pth`，对应服务器 `/home/yuhao2/Catch/catch_it_copy/best_model/throw_catch_best_reward_497.09.pth`。其 SHA256 暂为 null，因为本地没有该文件；加载时始终计算真实 hash，记录到检查报告和训练 checkpoint，恢复训练时比较实际 hash。服务器 check 后可把报告中的 hash 填入配置固定权重（应在开始正式训练前固定，恢复训练要求配置一致）。roll/bounce 继续使用原固定 hash。

新 throw 配置要求输出 `8+12=20`，其中底盘2、机械臂6、手指12，动作缩放沿当前主 PPO 配置 `[1.5, 0.025, 0.15]`，并在 check 中对照原 action2dict。未根据文件名推断实际形状：Teacher 加载时强制校验 expected_action_dim=20；误放旧 18 维模型会报错。教师仍需与实际训练的环境、动作缩放一致。

任务路由不变：throw 使用主 PPO/环境，roll 使用 645edc4，bounce Catch 使用 5fe75d5f。三个任务独立采样，等量 minibatch 更新同一个 53→20 MLP；不必修改 DAgger 算法或给学生增加维度。

同步代码到服务器，在 catch_it_copy 下先检查、再单独评估新 throw：

```bash
python -m distillation.run check --device cpu --output distillation/runs/three_teacher_check
python -m distillation.run teacher-eval --tasks throw --device cuda:2 --episodes 10 --output distillation/runs/throw497_teacher
```

先验证三任务 5 轮流程（roll/bounce 指标待校准，显式允许零基准）：

```bash
python -m distillation.run train --tasks throw roll bounce --device cuda:2 --iterations 5 --num-envs 2 --allow-zero-teacher-success --output distillation/runs/trb497_smoke
```

三个教师在各自对应环境中复现后，完整训练命令为：

```bash
python -m distillation.run train --tasks throw roll bounce --device cuda:2 --iterations 1000 --num-envs 2 --output distillation/runs/trb497_mlp
```

默认不写 --tasks 也会训练全部三个任务。以上正式命令保留零基准检查；指标未校准前不要把通过试训当成效果达标。每任务2个环境共6个采样进程，每次更新总 batch=768。请使用新输出目录：不能把此前两任务 resume 直接作为三任务续训，任务集合、缓存和配置都不同。历史交接文件关于旧18维 throw 的阻断已由本节取代，仅作为旧模型记录保留。

## 参数与日志

修改 JSON 后用 `--config path/to/config.json` 指定。建议先用默认纯学生执行；如果极早失败导致状态覆盖很差，可以尝试 `beta_start=0.5`、`beta_decay_iterations=100`，每环境整条动作随机选择教师/学生。最后必须留足 beta=0 阶段。该混合策略是可选项目参数。

输出包括：

- `resolved_config.json`、`teacher_check.json`、`teacher_baseline.json`。
- `metrics.jsonl`：每轮 imitation loss、beta、时间、缓存量、评估时最差任务相对教师的成功率差。
- `eval_*.json`：各任务成功数/总数、Wilson 95% 区间、回报、回合长度、终止原因、逐回合结果。
- `student_best.pth`：按“最差任务相对其教师的成功率差”选出；`student_last.pth` 每轮保存。均只含学生参数、归一化、任务和控制配置、来源信息，没有教师网络。
- `resume_last.pth`：每次评估或最终轮保存，额外含 optimizer、replay、RNG。默认 25 轮评估一次，所以故障续训可能损失最近不到 25 轮。

默认 roll 使用原评估优先使用的 `roll_eval_success`；bounce 使用环境 `info.success`。同时保存 `environment_success` 和 `legacy_truncated_rate`。旧 bounce PPO 测试把 `truncated` 当成功，和实际接球成功不等价，不能直接比较两个不同口径的百分比。保留原终止与奖励，不为统计修改环境。

当前未添加新的公共持球/掉球测量，不能声称已经验证这些指标；正式验收还应结合视频和持球质量。30 回合评估适合调试，最终至少每任务 300 回合，最好 3 个训练种子。达到每项比教师下降不超过 5 个百分点仅是候选目标。

## 验证

```bash
python -m unittest discover -s distillation/tests -v
```

包含真实 checkpoint 的严格恢复/推理一致性/冻结统计，以及动作掩码、分组损失、均衡缓存、回合历史重置、模拟环境下在线更新/保存/恢复/学生独立评估。这些测试不能替代 MuJoCo 闭环验证。

本机运行 `python -c "import mujoco"` 报 `WinError 1114`。请优先在原先能够正常运行三个教师的训练环境执行上面的检查与试训，不要把 DLL 报错当作蒸馏算法失败。
