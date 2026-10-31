# Basket 指定模型评估（2026-10-29）

模型：`/home/isee/Catch/catch_it_copy/outputs/BasketAimCarry/2026-10-29/01:23:05/nn/throw_basket_catch_best_reward_38.38.pth`。

种子 0–99，各局同时设置全局 NumPy、Python、Torch 和环境 Gym seed；CPU、确定性均值动作、原 PPO 归一化/动作缩放，严格加载全部权重，100 局全部正常结束。

| 指标 | 结果 |
|---|---:|
| 实际 basket_score / 主成功率 | 97/100 = 97% |
| Wilson 95%置信区间 | 91.55–98.97% |
| 几何命中篮圈 / 安全入篮 | 97 / 97 |
| 已确认真实出手 / 安全出手 | 100 / 100 |
| 危险机器人碰撞 | 0 |
| 平均策略回合仿真时间 | 0.646 s |
| 平均奖励 | 39.996 |

终止原因：`{'basket_score': 97, 'not_a_throw': 1, 'ball_on_floor': 2}`。

训练协议是 `basket_aim_v1 + fixed_base + fixed_physical + aim_carry_v1`。实际目标固定在 `[0.0, 1.05, 0.55]` m，篮圈半径 `0.2` m、球半径 `0.04` m、倾角 `25.0` 度；这是固定近目标投篮评估，不是随机远目标或移动底盘泛化评估。

成功需经过真实手部接触和确认自由飞行，球穿过扣除球半径后的有效倾斜篮圈开口，再观察0.3秒后无危险机器人碰撞。没有调用任何eval-only pose/hand/probe覆盖，固定手指参数来自训练本身。reset采用训练时的无外力物理稳定准备（0.470–0.528s），策略期间检查球actuator控制始终为0。

Carry前移6cm属于奖励目标，不是这版basket_score硬门槛。最后带球前移量中位数约 `0.0129` m，最大约 `0.0321` m；达到过6cm奖励目标的 `0/100` 局。近端手臂最后朝前速度≥.3m/s为 `55/100`。高投篮率不代表已学会明显的长距离带球前移。策略阶段手接触平均 `0.0267`s，不包括reset物理准备。

该训练目录未保存源码hash provenance，不能证明当前本地所有源文件与训练机历史文件逐字相同；模型的协议/动作布局/carry参数已与当前入口严格匹配，所有当前源码、模型和配置哈希见provenance.json。逐局记录见episodes.jsonl，复现脚本快照见EVALUATOR_SNAPSHOT.py。

正式统计仅来自本目录basket；相邻basket_stream_seeded_precheck为种子核查前的连续随机流预检查，basket_startup_failed为序列化启动失败且未产生评估局，不计入最终结果。
