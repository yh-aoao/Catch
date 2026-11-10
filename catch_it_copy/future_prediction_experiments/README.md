# Future 稳定训练与不确定性预测实验

两个独立实验，从头训练，不覆盖旧 Future、Contact、门控融合或固定相加版本。
未启动实际训练。已完成 CPU 合成检查，不代表已验证成功率提升。

| 版本 | 网络 | 学习率 |
|---|---|---|
| distillation_aux_future_stable | 与原 Future 完全相同，预测 1/3/5 策略步位移 | 1–5000 轮 3e-4；5000–10000 余弦下降至 3e-5 |
| distillation_aux_future_uncertain | 预测 1/3/5/8 策略步位移和每坐标方差 | 固定 3e-4 |

两者均保持原双相机深度、机器人状态、上一执行动作输入，没有任务ID、未来标签或物体真值推理输入。
当前没有增加拦截点目标、相对未来手部位置目标、Contact 或 PCGrad，保持比较范围明确。

## 训练命令

服务器先同步以下三个完整目录：
`future_prediction_experiments/`、`distillation_aux_future_stable/`、`distillation_aux_future_uncertain/`。
它们也依赖项目原有 distillation、distillation_visual_aux、visual_module_experiments 等公共代码和物理环境。

```bash
cd ~/Catch/catch_it_copy
conda activate dcmm

python -m distillation_aux_future_stable.run train \
  --output distillation_aux_future_stable/runs/catch_10000_seed123 \
  --device cuda:0 --gl-backend egl --egl-device-id 0 \
  --seed 123 --num-envs 8 --iterations 10000 \
  --eval-start 5000 --eval-every 1000 --episodes 300 \
  --skip-teacher-baseline
```

```bash
python -m distillation_aux_future_uncertain.run train \
  --output distillation_aux_future_uncertain/runs/catch_10000_seed123 \
  --device cuda:1 --gl-backend egl --egl-device-id 1 \
  --seed 123 --num-envs 8 --iterations 10000 \
  --eval-start 5000 --eval-every 1000 --episodes 300 \
  --skip-teacher-baseline
```

CUDA 与 EGL 编号按实际设备可见性调整。每任务 8 个环境，三任务共 24；同时运行两组为 48 环境。
默认 Adam、rollout_steps=64、updates_per_iteration=8、replay=8192 frames/task、bptt=16。
每次评估每任务 300 回合，三任务共 900。初次评估在5000轮，之后6000/7000/8000/9000/10000。
训练随机种子123；训练中固定验证随机种子20261110（独立于训练种子），两组完全一致。
验证集用于选模型，最终报告仍需另选未使用的种子测试，且统一并行/批量协议。
跳过教师闭环基线，仍加载、核验冻结教师，并用教师动作作蒸馏标签。

每次验证独立保留 student_005000.pth 等权重；student_last 每轮保存；resume_last 每200轮保存优化器/replay/RNG。
student_best 的选择规则仍为三任务最低成功率，不是平均成功率。

## 预测增强版定义

- mu：未来球心世界坐标减当前球心世界坐标，单位米，四个时间点共12维。
- 每坐标 logvar：归一化误差 `(mu-target)/0.1m` 的对数方差，通过 sigmoid 限定在[-6,4]。
- 初始 sigma=0.1m，实际 sigma 范围约[0.005m,0.739m]。这是学习的不确定性，未证明校准。
- reliability = 1/(1+sigma/0.1m)。将 `[mu*reliability, sigma]` 共24维经 Linear 映射回256维，与GRU特征残差相加后输出动作。
- sigma 在动作融合路径 detach；动作模仿损失不直接调整不确定性头参数，NLL仍会更新该头和共享主干。
- 未来监督使用有界对角高斯负对数似然 NLL，包含 logvar 惩罚项；不能简单通过增大方差无限降低损失。
- 总损失 = 原模仿损失 + 0.1 位置SmoothL1 + 0.1 归一化未来NLL。速度损失权重保持0。
- NLL可能为负；其数值与旧Future的SmoothL1不可直接比较，优先看米制RMSE、覆盖率与闭环成功率。
- 采用原有效区间约定：首次手部接触前预测；保留桌面反弹；不跨回合或越过终止位置。
- 增强版改变了时间点、概率目标和融合编码，是一项联合结构改进；若有效，需进一步拆分消融才能归因到单个因素。

## 诊断与解释边界

每100轮及评估轮次，在 metrics.jsonl 中记录 replay训练批上的诊断（配置 diagnostics_every 可修改）：

- diagnostic/{throw,roll,bounce}/imitation_mean
- 各时间点 /rmse_m、/count（有效预测标签数量）
- 增强版 /sigma_m_mean、/coverage95_mean（各坐标边际95%区间覆盖率，不是三维联合覆盖率）
- Bounce 阶段：pre_rebound_proxy、post_rebound_proxy、near_hand、post_hand_contact。

阶段来自整回合历史世界高度差的下降→上升反向、相对位置距离0.15m阈值、过去实际手部接触。
反弹阶段只是运动学代理：可受数值噪声、运动轨迹影响；不是直接桌面碰撞标签。阶段有优先级：已接触 > 近手 > 已观察反向 > 未观察反向。
这些标签只用于报告，绝不进入网络 forward 或损失权重。
没有有效样本时 mean/rmse 报0占位，必须同时看 frames/count；不是误差为零。
每轮诊断是8次优化批的均值，不是固定独立验证集指标。
原始 *_sum 和 count 可用于多轮汇总；不要直接把无样本的0占位当作有效值平均。

## 中断续训

原命令追加对应的 `--resume <同一输出目录>/resume_last.pth`。
不能直接 resume 旧Future/Contact模型；新schema区分实验。稳定版虽结构相同，此入口仍要求新schema。
学习率由已恢复 iteration 和固定 start/end 配置计算，不依赖缺失的scheduler状态；更改目标iterations不会自动移动衰减终点10000。
不要在训练途中更改 horizons、方差界、归一化尺度或衰减设置；更改配置应新开实验。
源码修改经核查后才使用 `--allow-source-changes`，改动会保存审计记录。

## 验证

```bash
python -m unittest future_prediction_experiments.test_future
```

7项CPU合成测试：原Future权重/前向精确一致、可变horizon/接触边界/恢复、概率头梯度与mask、
学习率边界、NLL分段一致/无真值推理泄露、默认配置时间表、Catch及Track完整保存加载续训。
尚无真实MuJoCo/GPU smoke或成功率结果。原有28个源码文件SHA256未变化。
