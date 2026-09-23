# Roll 评估与训练解耦

## 2026-09-23 修正

Catch 首次完全越过桌沿且无桌地接触后，记录 departed=True。此后回到桌沿下方不再仅因位置清零；真实桌地接触、速度超限、离手区域和过长接触间隙仍会打断保持。Track 的桌沿外拦截口径不变。

终局 checks 新增 max_duration（本回合最长有效保持秒数）、first_contact_time（首次手部接触的累计物理时间）、reset_counts / last_reset_reason。计数仅记录有保持进度被清零的事件。outside_edge、departed、table_contact、floor_contact 分别显示位置与真实接触，避免混淆。

Catch 记录滤波前手指动作的 raw_delta_rms_max、raw_sign_flips（幅值超过 0.001 rad 的反向分量次数）、action_samples，以及滤波后 applied_rms_max。这些是物理缩放后的目标增量统计，并非网络输出或实际关节振动的直接测量；只能为策略抖动提供线索，不能单凭它们排除控制器问题。

本次不修改奖励、控制器、动作滤波、时限和终止条件。用户认为抖动来自策略，后续优先依据诊断检查策略接触后反复开合的问题。

## 评估口径

`mean_success` 现在优先统计 `roll_eval_success`：

- Track：episode 内出现真实手掌/手指接触，球完全越过桌沿且未接触桌面/地面。无需保持低速；曾拦截后掉落仍属于成功拦截，不代表 Catch 成功。
- Catch：相对手速不超过 0.15 m/s，球心距掌中心法向偏移的接球参考点不超过 0.16 m，球已离桌且未碰桌地，真实手部接触累计保持 0.30 s。
- 允许低速且仍在手部区域内的接触间隙不超过 0.04 s。间隙不累计保持时间；超限/超速/离手区域/碰桌地则重新计时。
- `roll_final_hold`：回合最后一个物理步仍有真实手部接触并达到上述保持时长。短接触间隙在终局也不会计为保持。

阈值在 `gym_dcmm/utils/roll_evaluation.py`，是待实际轨迹校准的评估参数。所有接触使用手掌和手指几何集合，不通过 geom 编号大小猜测，也不把手腕/机械臂触球算作手部接触。

## 不改变训练行为

观察函数仅在每个物理子步读状态，评估发布在原奖励与终止逻辑之后。保持当前奖励、成功终止、失败终止、时限、动作、初始化、PPO 优化与按 reward 保存最佳模型的逻辑。

Track 同时补齐原来遗漏的 `info['success'] = step_touch and not terminated`；新的手指拦截指标另存 `roll_eval_success`，不会令手指接触提前结束。Catch 的 `info['success']` 仍为旧标准，`roll_legacy_success` 保留对照。

因此稳定接住过但后来掉球可显示 eval_success=1、final_hold=0；超时与新评估成功也可能同时出现，表示截止前完成了评估目标，而原终止条件没有改变。旧终止条件若过早结束，仍可能阻止累计 0.30 s，本次没有消除这一限制。

Roll Track 跟踪参考点仍为 link6，没有改成掌心；新指标使用掌心区域只是评估，不改变策略输入或奖励。

## 输出与使用

原训练/测试命令无需变更。增加 `roll_log=true` 查看每回合 `[roll-eval]`：旧标准、新标准、最终保持、原结束原因、接触状态/速度/保持时长及未满足项。unmet 是终局检查，并非整个 episode 历史。

三个 Roll PPO 入口均读取自动重置的 final_info。`[roll-summary]` 在第 1 回合及每 20 回合输出累计 eval_success、final_hold、legacy_success；mean_success 本身沿用各 PPO 原统计窗口，可能与累计值略有不同。

可直接测试旧模型，不要求重新训练。不能把新成功率与旧标准的数字直接比较。

验证：5 项评估测试、3 项历史行为/路由测试通过；Catch 除评估调用与 reset 清理外的类 AST 与修改前一致，修改的 Python 文件通过语法检查。未运行完整 MuJoCo 实际模型测试。
