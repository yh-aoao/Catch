# Basket 任务完整交接（2026-10-26）

## 给接手助手的说明

这是项目当前状态和历史证据，不是要求自动执行所有后续计划。请先核对实际代码、用户最新要求和服务器输出。不要把离线单元测试当作真实仿真训练验证，不要把“抛出成功”当作“入框成功”。

当前目标：固定底盘，从零训练机械臂和灵巧手，把小球抛进近距离目标入口。使用独立奖励版本 `basket_aim_v1`。用户明确要求从零开始，不是在旧模型上继续训练；必须保留向前飞行距离奖励。暂时不做移动底盘，也不做视觉蒸馏。

交接时本地 Git HEAD 为 `39ef02d`（此文档尚未包含在该提交中）。Windows 工作区 `F:\yjs\Catch`，实际项目 `catch_it_copy`。常用 Linux 路径 `~/Catch/catch_it_copy`，conda 环境 `dcmm`。服务器 Python 有3.8/3.10两种，工具需兼容。另一台电脑须同步代码和权重，本文档不是代码包。

## 1. 当前接口与任务边界

- 入口：`train_DCMM.py`，Hydra `configs/config.yaml`。
- `object_motion=throw_basket`；`task=Catching_OneStage`。
- `basket_fixed_base_training=true`，`basket_arm_control=joint_velocity`。
- `basket_reward_version=basket_aim_v1`，`basket_curriculum_stage=3`。
- 固定底盘策略输出18维：机械臂6维、手12维；不输出底盘动作。
- 机械臂动作反归一化比例 .025，入口检查该比例；不能拿旧Cartesian控制权重按joint_velocity解释。
- 策略输入35维（状态输入，不是相机输入）。包括底盘速度、末端位置/姿态/速度、物体位置/速度、手部状态，以及篮筐相对位置3维和停车目标相对位置2维。实际拼接以 `PPO_Catch_OneStage.obs2tensor` 为准。
- 现有Actor通常为MLP [256,128]，价值网络分离；以训练resolved_config为准。
- 目标是按规定方向穿过倾斜圆形入口。不是实体篮筐内稳定停留；后者需要额外的容器碰撞与成功标准。
- 原PPO通常按平均奖励保存best_reward，不是按安全入框率选best。评估时应检查安全入框率，不能仅凭文件名判断最优。

## 2. 当前近目标与奖励

独立配置函数：`gym_dcmm/utils/basket_aim.py::make_config`，复制基础配置，避免修改旧任务的全局参数。

| 参数 | 当前值 |
|---|---|
| 目标中心（世界坐标，米） | [0, 1.05, 0.55] |
| 入口半径 | 0.20 m，沿用基础配置 |
| 入口倾角 | 25度，沿用基础配置 |
| 球半径 | 0.04 m |
| 距离奖励上限 | 0.5 m |
| 有效释放 | 最后手接触时向目标水平速度>=0.3m/s，随后无接触>=0.05s |
| 入框最小自由飞行 | >=0.1s 且水平位移>=0.2m |
| 命中后安全观察 | 0.3s |
| 瞄准误差尺度 | sigma=0.25m |

近目标根据旧策略约0.7～1m的前向飞行表现提出，尚无本对话内实际训练结果证明它是最优位置。目标固定在世界坐标系，不随手/球运动。重置时更新basket_target模型位置。当前保留原入口半径，没有自动扩框课程。

| 奖励项 | 定义 |
|---|---|
| 有效抛出 | +5，每回合一次 |
| 前向自由飞行距离 | 最大距离增量累计最多+5，0.5m封顶 |
| 目标精度 | 10*exp(-(e/0.25)^2)，有效释放后一次 |
| 实际命中 | +20，一次 |
| 未有效释放失败 | 结束/超时仍未释放，-5一次 |
| 危险碰撞 | -40一次，底盘碰撞/连杆自碰撞共用预算 |
| 机械臂动作变化 | -0.01*mean((a_t-a_prev)^2)，归一化动作；无手指，首步不扣 |

瞄准误差e：使用最后手接触时的位置、速度以及重力，求未来2秒内轨迹穿过入口平面的正侧到负侧交点，到入口中心的平面内距离。不存在合法交点时该项为0。不是每步奖励，也不在持球时刷接近奖励。

距离奖励仅奖励历史最远距离的增量。落地滚动、再次接触手或其他物体后的移动不计。
总正奖励理论上最多40，碰撞扣40（另有平滑惩罚）；未释放失败可能叠加。

## 3. 成功与终止

- 实际穿越要求：球轨迹从入口正侧到负侧，球心穿越半径<=入口半径-球半径。
- 必须有效释放，满足最小时间/水平位移；释放后再接触会阻止首次有效命中。
- 有效命中记录后继续观察0.3s；这期间球落地不提前结束，但机器人危险碰撞仍终止并扣分。
- 安全观察结束后设结束原因 `basket_score`；超时或碰撞先发生则不算安全成功。
- `basket_hit`：曾有效穿过入口。
- `launch_success`：满足有效释放。
- `safe_launch_success`：回合结束时有效释放，且无当前定义的机器人危险碰撞。
- `safe_basket_success`、`success`、`basket_success`：该版本中完成安全观察的实际入框。
- 原 PPO `mean_success` 对此版本是安全入框结果；对旧launch版本仍只表示严格入框，不能当作抛出率。
- 常见结束原因：`ball_on_floor`、`base_collision`、`arm_self_collision`、`not_a_throw`、`basket_score`、`timeout`。

## 4. 自碰撞修复历史和实现范围

旧策略有明确仿真自碰撞，真实contact日志记录link1↔link4、link1↔link5，采样法向力最高分别约335N、495N；另有指尖↔base_link接触。不能解释成所有关节都发生碰撞。

现实现：
- 每物理子步检查link1～link6非相邻连杆实际接触，接触距离<=0且法向力>0.1N时立即终止。
- 阈值0.1N只是初始去噪值，尚未系统标定，不是硬件安全标准。
- 机械臂/手撞指定底盘几何也受罚；球撞底盘不算机器人危险碰撞，但仍保持原终止。
- 不覆盖所有手指自碰撞、所有手臂-手组合或全部邻接关系；“安全”只相对当前检测范围。
- 训练保护不依赖 `basket_self_collision_debug`。
- 诊断开关另检查几何距离，`geometry_proximity`只是接近/代理重叠，不能直接等同实际承力接触；碰撞网格可能是凸包近似。

## 5. 历史版本，不要混用

| 版本 | 用途 |
|---|---|
| legacy / launch_v4 / launch_v5 | 早期实现，奖励较复杂，保留兼容 |
| launch_simple_v1（早期） | +5释放、最多+10距离、未释放-5 |
| launch_simple_v1 safety_v2 | 增加连杆自碰撞、统一-20碰撞罚分、小权重动作变化 |
| basket_aim_v1（当前） | 独立近目标，保留距离项并新增瞄准/命中奖励，碰撞-40 |

旧版原目标中心为[0,2.2,0.9]，近目标不能用旧释放距离/飞行位移阈值直接判成功。
用户曾询问加入移动：当前先固定底盘入框。后续可做移动到位→停稳→投掷；不能只关闭fixed_base就将18维旧策略当成20维移动策略。

## 6. 从零训练命令

在Linux的catch_it_copy目录：

```bash
conda activate dcmm
cd ~/Catch/catch_it_copy
python3 train_DCMM.py \
  test=False task=Catching_OneStage num_envs=32 \
  vector_mode=async \
  object_motion=throw_basket \
  basket_fixed_base_training=true \
  basket_arm_control=joint_velocity \
  basket_reward_version=basket_aim_v1 \
  basket_curriculum_stage=3 \
  basket_log=false basket_self_collision_debug=false \
  device_id=0 output_name=BasketAimV1_FromScratch
```

不要传checkpoint_tracking/checkpoint_catching，不是resume旧模型。device_id改成服务器可用卡。
num_envs=32是该单任务32个并行环境，不是蒸馏中“每任务32个”。
输出位于outputs/output_name/日期/时间等子目录，权重在对应nn目录，查实际生成文件，不要猜具体best_reward文件名。

## 7. 测试命令

```bash
python3 train_DCMM.py \
  test=True task=Catching_OneStage num_envs=1 vector_mode=sync \
  object_motion=throw_basket \
  basket_fixed_base_training=true \
  basket_arm_control=joint_velocity \
  basket_reward_version=basket_aim_v1 \
  basket_curriculum_stage=3 \
  basket_log=true basket_self_collision_debug=true \
  viewer=true basket_viewer_speed=0.25 \
  device_id=0 checkpoint_catching=/绝对路径/模型.pth \
  output_name=BasketAimV1_Test
```

速度0.25为四分之一速，1为实时，0不限速。只有单环境fixed Basket viewer测试限速；不改物理步长，不影响训练。大规模测试关闭距离诊断以避免开销。训练/测试必须同版本目标和控制语义。
关注 `[basket-end]`、`[basket-throw]`、`[basket-self]`。Ctrl+C后的渲染清理异常与训练过程碰撞不是同一问题。

## 8. 已有证据与验证边界

旧抛出视频和日志（仅本机文件，不保证另一台电脑存在）：
- C:/Users/29491/Downloads/10_26.webm
- C:/Users/29491/Downloads/temp.txt

统计28个完整回合：全部launch_success，全部ball_on_floor结束；前向距离0.717～0.974m；未记录底盘碰撞/连杆contact/proximity，入框0/28。视频姿态较旧自撞版本明显改善。
重要：该日志缺少safety_v2、arm_smoothness、safe_launch_success等字段，测试端疑似仍为旧代码，不能据此认定最新安全奖励效果；不能把28回合推广为稳定100%安全率。

basket_aim_v1：此前35项相关单元测试与语法检查通过；包含4项aim测试、9项simple测试、22项catching测试。未完成本地真实MuJoCo训练验证，测试不是完整动力学集成验收。当前尚未收到近目标版本训练后的实际视频/成功率。
建议下一步先确认目标可视位置与成功穿越逻辑，再用约200回合多个种子统计安全入框、释放、碰撞、预测误差、奖励各项。不要直接增加移动自由度。

## 9. 关键文件

| 文件（相对catch_it_copy） | 作用 |
|---|---|
| train_DCMM.py | CLI、控制接口检查、sync/async选择 |
| configs/config.yaml | Hydra开关、默认参数 |
| configs/env/DcmmCfg.py | 原Basket参数、simple默认奖励/安全阈值 |
| gym_dcmm/envs/DcmmVecEnv.py | 目标重置、物理步接触、命中、奖励接线、日志 |
| gym_dcmm/utils/basket_simple.py | 释放/距离奖励、自碰撞实际contact筛选 |
| gym_dcmm/utils/basket_aim.py | 近目标配置、弹道交点、aim/命中奖励 |
| gym_dcmm/utils/basket_diagnostics.py | 只读连杆诊断 |
| gym_dcmm/utils/worker_errors.py | 原始子进程异常打印，保留抛出 |
| gym_dcmm/algs/ppo_dcmm/ppo_dcmm_catch_one_stage.py | 18维策略、35维观测、测试限速、权重保存 |
| tests/test_basket_aim.py | 交点、奖励一次性、配置隔离 |
| tests/test_basket_simple.py | 释放/距离/安全/平滑，以及真实奖励函数AST执行 |
| BASKET_AIM_V1.md / BASKET_LAUNCH_SIMPLE_V1.md | 各版本补充说明 |

## 10. 已修复问题和服务器操作注意

1. 曾在compute_reward(self,obs,info,ctrl)中误用action['arm']，首步NameError。已改ctrl['arm']，并补真实奖励分支测试。如果重现说明服务器代码未更新。
2. Gymnasium旧experimental AsyncVectorEnv可能先拆包None再报告worker错误，表现cannot unpack non-iterable NoneType。已有worker_errors打印完整异常；可将vector_mode=sync诊断，保持num_envs=32避免改变PPO批量约束。
3. 服务器Git有独立本地提交、未提交修改与runs/。必须检查git status；不要git clean、不要盲目reset或git add .。合并有冲突时先解决<<<<<<<等标记再运行。此前最新一次服务器pull提示Merge made by ort表示合并成功，原未提交修改仍保留。
4. 用户明确有必要保留服务器上的某些蒸馏run.py修改。不要为了更新Basket覆盖无关修改。
5. 集群曾因主机RAM/swap压力SIGTERM蒸馏任务，不是本Basket奖励报错。不要混为GPU显存问题。
6. 本文不提供精确训练轮数/耗时承诺，PPO训练预算以实际configs/train和resolved_config为准。

## 11. 可粘贴到新对话的起始请求

请先阅读 BASKET_HANDOFF_2026-10-26.md 并核对实际代码。我们当前在做 basket_aim_v1：固定底盘，从零训练18维机械臂+手策略，把小球抛过近目标入口；保留距离奖励、一次性瞄准/入框奖励和自碰撞保护，不加载旧模型，不加入移动。请结合我接下来提供的训练日志和视频继续检查。不要将旧launch_simple_v1的抛出成功率当成入框率，也不要覆盖服务器上无关的本地修改。
