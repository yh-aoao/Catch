# Throw / Roll / Bounce 多教师蒸馏实施与交接说明

日期：2026-10-02。本文是实施方案，蒸馏代码尚未实现。
核对时 HEAD：d2e77125983acdd1707bad0ec9afb613474a68d5。新对话开始前重新核对 git status、分支和当前代码，保留用户修改。

## 1. 用户决策

用户报告 throw、roll、bounce 三个运动模型已经训练得比较好，下一步参考 Robot Parkour Learning，把三项技能蒸馏到一个模型。研究主线仍是多任务移动操作，Basket 暂不加入。

默认目标为完整 Catching（移动跟踪与抓取）策略的整合。实施前确认三个 checkpoint 的任务类型，不能混用某任务 Tracking 与另一任务 Catching 作为同级教师。保持原教师环境的奖励、控制器、初始化、终止、动力学和时限，不在蒸馏期间同时修改这些逻辑。

## 2. 三个模型是否足够

仅三个权重文件不够，每个教师必须具备完整的推理和环境配置：

- 当前最优 checkpoint 路径及 SHA256、任务类型、模型结构。
- 观测归一化统计；TwoStage 常用 running_mean_std_track 和 running_mean_std_hand。缺失时不能默默用初始统计代替。
- 能复现效果的完整测试命令、resolved_config.yaml 或原训练配置、准确的环境与控制代码版本及模型资产。
- 球物理参数、发射分布、随机化范围、控制频率、动作缩放、动作滤波、成功和终止标准。
- 可以运行的 MuJoCo 环境：DAgger 需要学生实际交互和教师在线标注。只有静态演示可以做行为克隆，但不是完整在线 DAgger。
- 每项教师的基准成功率、评估种子和持球表现，不能根据模型文件名的 reward 判断成功率。

完整 TwoStage Catch checkpoint 通常已包含跟踪和抓取分支，不一定需要额外 Track checkpoint。本仓库 Roll TwoStage.save 保存整个 model 及两组归一化统计；load_tracking_model 在提供 checkpoint_catching 时跳过独立 Track 加载。但必须逐个检查实际文件，不能保证所有历史 checkpoint 相同。OneStage 的归一化键可能不同。

用户待提供：

| 任务 | 最优 Catch checkpoint | 原测试命令 | OneStage/TwoStage 与配置 | 基准成功率 |
|---|---|---|---|---|
| throw（用户已确认） | best_model/catch_two_stage.pth | 待提供 | TwoStage 结构，动作头 6+12 | 待验证 |
| roll | best_model/roll_catch_best_reward_252.54.pth | 待提供 | TwoStage 结构，动作头 8+12 | 待验证 |
| bounce | best_model/bounce_catch_best_reward_653.87.pth | 待提供 | TwoStage 结构，动作头 8+12 | 待验证 |

历史出现过 bounce_catch_653.87.pth，不要自动认定为现在的最优教师。

## 3. 论文方法与本项目改编

Robot Parkour Learning（CoRL 2023）§3.2 用 DAgger 将五个技能教师蒸馏到单个视觉学生。训练依据地形为学生访问状态选择对应教师；学生使用本体感知、上一动作和深度图编码，网络带 GRU。原文对 tanh 输出使用 binary cross entropy 形式的差异损失。

原始来源：
- 论文：https://robot-parkour.github.io/resources/Robot_Parkour_Learning.pdf
- 正式出版页：https://proceedings.mlr.press/v229/zhuang23a.html
- 作者项目页：https://robot-parkour.github.io/

本项目借鉴的是“学生交互、对应教师在线标注”的多教师 DAgger，不是平均网络权重。以下任务标签、MSE、行为克隆预热和参数均为项目建议，不是论文原配置。

首版先做状态输入、显式任务标签的统一学生。先验证整合能力，再研究无标签、历史/GRU、视觉、真机和物理参数泛化。显式标签方案不能宣称自主识别运动类型。

## 4. 当前代码入口与陷阱

工作目录 catch_it_copy，先读根目录和子项目 CLAUDE.md。

| 任务 | 当前 Catch PPO 家族 | 环境入口 |
|---|---|---|
| throw | gym_dcmm/algs/ppo_dcmm | DcmmVecEnv.py 主类 |
| roll | gym_dcmm/algs/ppo_dcmm_645edc4 | 主类 __new__ → roll_compat.make_roll_645edc4 → DcmmVecEnv_roll_645edc4.py |
| bounce | gym_dcmm/algs/ppo_dcmm_5fe75d5f | 主类 __new__ → BounceEnv |

train_DCMM.py 决定 agent 路由，DcmmVecEnv.__new__ 决定环境路由。Bounce Tracking 另走 ppo_dcmm_july22，不能误用于 Bounce Catch。Roll Tracking 是主文件 RollTrackingEnv 的 07c176f 逻辑。旧模型需要匹配其原始代码，当前路由不保证与训练时一致。

逐个核对 obs2tensor、model_act、act_inference、action2dict、环境 step 和 normalizer。某些 restore 函数允许部分加载、扩展或跳过不匹配张量；教师必须严格恢复，任何 missing/unexpected/shape mismatch 都应报错。

Roll 有环境内手指目标平滑及缓存。相同网络输出在不同控制器下未必产生相同行为。首版保留每个任务的执行适配器，明确报告“单策略网络＋现有任务控制适配器”。不要宣称已经统一底层控制。后续合并控制器须单独评估。

## 5. 学生输入输出契约

第一版：一个共享 MLP、一个共享动作输出头，加入 throw/roll/bounce 三维 one-hot。部署只加载一个学生，不加载三个教师；暂不需要 MoE。

当前 TwoStage 拼接常见字段：底座平面速度 2；末端位置 3、四元数 4、线速度 3；球位置 3、线速度 3；手状态 12，合计 30。这是代码字段计数，必须运行时核对所有教师的形状、语义、坐标系、单位、四元数顺序及手关节排列，不能依赖过时注释。

建议学生输入：统一的上述状态＋上一归一化执行命令 20＋任务标签 3。如果字段核实无误则为 53 维。上一动作必须是实际选中执行的命令，teacher/student 混合采样时不要存未执行的学生预测。环境滤波隐藏状态先保留；若可观测性不足，增加目标关节状态或历史/GRU。

教师各自使用原观测处理和冻结 normalizer；学生从三任务均衡数据估计自己的独立 normalizer，第一版固定它，评估期间不更新。不假设所有 pos3d 是世界坐标，按实际观测代码转换。

预期输出 20 维：底座 2、机械臂末端增量 6、手指命令 12。先核对三个动作缩放、关节映射和控制周期。若一致，可用原归一化命令作标签；若不一致，先映射到公共物理命令，再按公共尺度归一化。不能直接平均含义不同的列。

使用确定性教师 inference 输出，经原逻辑裁剪后作标签；不采样 PPO 探索噪声。学生使用有界 tanh 输出。模型中两阶段分支可以作为一个完整教师查询，不需要分别蒸馏后再拼接。

## 6. 实施流程

### A. 教师复现

逐个恢复教师，冻结参数，model.eval、normalizer.eval。TeacherAdapter 与原 test 推理在同一批观测上的输出应一致，FP32 建议误差小于 1e-5；再比较缩放后动作与闭环表现。先每项至少 100 回合粗测，记录固定种子和参数。

适配器只构建推理必要模块，不调用 PPO.train，不重复推进仿真。若有 recurrent 状态，每个 done 按环境索引重置；教师状态也沿学生实际历史更新。若不能复现教师，先解决版本/配置问题，不开始蒸馏。

### B. 行为克隆预热

分别采集 student_obs、teacher_action、task_id、episode_id、done。起始预算可每任务约 10 万 transitions，按实际覆盖调整，同时记录完整回合数。训练验证按完整 episode/种子分割，禁止相邻帧随机拆分造成泄漏。

建议初始 MLP [256,256,128]、Adam 3e-4；均为待调实验参数。三任务均衡采样；底座、机械臂、手三组分别计算平均 MSE，再等权求和，防止 12 维手指压过 2 维底座：

L = mean_over_tasks[(MSE_base + MSE_arm + MSE_hand)/3]。

不蒸馏 critic，不合并教师 PPO 奖励，不需旧优化器。MSE 是项目连续动作基线；KL 或论文 BCE 不能在不了解输出分布和尺度时直接套用。

### C. 在线 DAgger

各任务独立环境池，首版可每任务 8 个环境，总计 24，依算力调整；蒸馏不沿用 PPO minibatch 乘积限制。任务在一个 episode 内固定。

```python
for iteration in range(iterations):
    for task, pool in task_pools.items():
        obs = current_obs[task]
        x = student_adapter(obs, previous_executed_action, task)
        with no_grad():
            label = teachers[task].predict(obs)
            prediction = student(x)
        replay.add(x, label, task, episode_id)
        command = label if bernoulli(beta) else prediction
        next_obs, reward, terminated, truncated, info = pool.step(action_adapter(command))
        # 更新 current_obs、上一实际执行命令，按 done 清空每环境历史/隐藏状态
    train_student(task_balanced_batches(replay))
    evaluate_student_only()
```

教师必须标注学生当前访问的状态，不是另一条教师轨迹。beta 可从 1 降到 0.5、0.2、0，按闭环表现调整；这不是论文原参数。必须有充分 beta=0 的学生采样，不能始终让教师执行。每环境选择完整教师/学生动作，不逐维拼接。

Buffer 按任务分配容量，保留演示与学生状态，防止某一任务被覆盖。教师在严重偏离状态上也可能失效，DAgger 不能创造教师本身不具备的能力。自动 reset 时区分 final_observation/final_info 和新回合初始状态，避免跨回合错误配对。

### D. 验收与对照

beta=0、只加载学生，分别评估三个任务。最终建议每任务至少 300 回合，记录成功数/总数和置信区间；训练建议 3 个种子。可讨论的初始验收标准：每项较教师下降不超过 5 个百分点，且持球时长、掉球率不明显变差。该门槛是建议，不是用户已确认指标。

保留每任务原成功指标用于配对比较，再增加只读公共持球指标；不得为了统一统计改变教师环境终止条件。不能只报告平均值掩盖单任务退化。

必要对照：三个独立教师、BC 学生、BC+DAgger 学生。后续再做无标签、MLP/GRU、直接多任务 PPO。先同分布整合再测试泛化，不把使用既有 DAgger 本身当作论文创新。

## 7. 建议文件与输出（尚未实现）

- configs/distill_three_tasks.yaml：教师清单、固定环境配置、采样与学生参数。
- gym_dcmm/distillation/teachers.py：严格恢复和教师适配。
- observations.py、actions.py：语义契约与映射。
- student.py、buffer.py、dagger.py。
- distill_DCMM.py：teacher_check、collect、bc、dagger 入口。
- evaluate_distilled.py：student-only 评估。
- tests/test_distillation_interfaces.py：动作一致性、归一化冻结、done/reset、均衡采样、教师无梯度。

新 checkpoint 保存学生参数、normalizer、obs/action schema 版本、任务编码、控制适配配置、教师 hash 与代码版本、训练进度；可恢复训练包另存 optimizer、随机状态及 buffer 信息。部署包不嵌入教师网络。

以上是建议路径，不要把它们当作已经可运行的命令。

## 8. 新对话接手顺序

1. 读本文件、CLAUDE.md，核对工作区。
2. 收集三个当前最优模型路径和可复现测试命令；可同时实现无模型依赖的骨架。
3. 审计 checkpoint 和归一化，验证教师推理一致性和闭环表现。
4. 实现显式任务标签的状态输入 BC，再接在线 DAgger；保持原任务逻辑。
5. 输出单学生 checkpoint 和三任务对照结果，再讨论无标签/视觉/真机。

本机此前 MuJoCo 导入报 WinError 1114，实施时重新检查。CPU 接口测试不等于仿真闭环或训练成功。已读取用户 best_model 下三个 checkpoint 的键及张量形状，见下方补充；尚未完成教师闭环验证。


## 9. 用户提供的实际模型（2026-10-02 补充，优先于此前的维度预估）

根目录：F:\yjs\Catch\catch_it_copy\best_model。用户确认三个训练好的模型放在此目录。已通过 torch.load(weights_only=True, map_location='cpu') 检查，未修改模型。

| 文件 | 跟踪头 mu_t | 抓取头 mu_c | 跟踪/手部归一化维度 | SHA256 |
|---|---|---|---|---|
| bounce_catch_best_reward_653.87.pth | 8×128 | 12×128 | 18 / 12 | f3ff1fe5aa34a8d4384dda6915b6749847002447d2816f18e72b661c6254a09c |
| roll_catch_best_reward_252.54.pth | 8×128 | 12×128 | 18 / 12 | 40020e1b13a29ad49de9e9b6ad2e6316678f418bdd8712be898b7a257af3d0ba |
| catch_two_stage.pth | 6×128 | 12×128 | 18 / 12 | 81585772adf9b368e299875a3dce29247e23c641d8294295bec07dfb6450aaf6 |

三者顶层键均为 model、running_mean_std_track、running_mean_std_hand、value_mean_std，具备 TwoStage 参数与归一化结构。不能仅凭键与维度证明闭环效果已复现或训练版本已匹配。

重要：用户已确认 catch_two_stage.pth 为 Throw 教师。其动作头共 18 维，Roll/Bounce 共 20 维，因此第 5 节的统一 20 维只能作为目标接口，不能直接套用。必须先找到该模型匹配的网络、原 action2dict 和环境：6 维跟踪输出究竟包含哪些底座/机械臂命令、是否缺失姿态维度，当前尚未确认。禁止按猜测在尾部补两个零或部分加载到 8 维跟踪头，否则会错置手指动作。明确语义后逐字段映射；教师不控制的通道应有显式默认值与 loss mask，并验证适配后的实际执行动作。

下一对话无需再次询问三个文件存放位置；catch_two_stage.pth 的 Throw 任务归属已由用户确认，无需重复询问；仍需收集每个模型可复现效果的完整测试命令及配置。优先做这个 18/20 维接口差异的审计，再开始数据采集。

