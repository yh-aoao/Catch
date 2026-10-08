# Throw / Roll / Bounce：共享 MLP 在线 DAgger

实现日期：2026-10-02。第一版使用一个学生 MLP `[256,256,128]`，直接在线 DAgger，默认 `beta_start=0`，不包含离线 BC 预热、PPO 微调或 GRU。只训练学生，不更新教师和原环境。

**当前状态：训练链路和 CPU 测试已实现；本机 MuJoCo 导入仍报 WinError 1114，尚未完成真实仿真闭环验证。roll/bounce 可在原先能正常训练的机器上先验证。三任务入口因 throw 18D 动作映射未确认而有意停止，不能宣称三任务已经可训练。**


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

当前期望：三个教师严格恢复、观测和推理对照通过；roll/bounce 动作转换对照通过；throw 显示 `interface_ready=false`，进程非零退出。报告在 `teacher_check.json`，这个退出不是训练代码崩溃。

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

## 三任务训练尚需解决的 throw 接口

配置为 `configs/three_tasks.json`。三个模型位置和 SHA256 已填好，无需重新提供路径。

throw 输出为 `6+12=18`，roll/bounce 为 `8+12=20`。当前代码的机械臂动作是 xyz 位移＋zxy 三个旋转增量，但旧 throw 的四个机械臂通道具体定义未核实。**禁止尾部补两个零，禁止部分恢复到 8D 跟踪头。**

必须找到训练该 throw checkpoint 时的 `action2dict()`、`move_ee_pose()` 及环境配置，并复现原测试效果。用户目前提供的是 `test=False ... checkpoint_tracking=... object_motion=throw/roll` 训练模板，无法识别这个旧模型的执行接口。

确认后，`action_indices` 按“旧输出每一列对应哪个公共动作通道”填写，手指必须映射到 8..19；不受控通道设置明确默认值并屏蔽监督和执行。若旧旋转参数不能简单逐列映射，必须扩展适配器，不能只填索引。必要时通过 `env_factory` 使用经验证的兼容环境（该工厂对外应提供 20D 动作/30D 状态接口）。只填写 mapping 并不能证明动力学兼容。

三个教师在各自对应环境中复现后，完整训练命令为：

```bash
python -m distillation.run train --device cuda:0 --output distillation/runs/three_tasks_mlp
```

**此命令在当前默认配置下会提前报 unresolved action mapping；目前不要用它启动长训练。** 两任务学生仍保留三维任务标签，但未训练 throw，不能用作 throw 策略，也不能直接作为三任务 resume。

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
