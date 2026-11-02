# 2026-11-02 Roll / Bounce Catch三球测试

每个条件100回合，种子0–99，原PPO确定性推理，严格加载actor及归一化参数。仅评估，不修改训练环境。
配置缺失：11-2只有权重；按此前equal-v2命令重建配置。若服务器自行覆盖过发射、速度、控制或奖励选项，需要提供原resolved_config.yaml复核。
Roll：region_retention_v5，最终区域保持并排除失败；稳定持球是另列的曾达到低相对速度0.3秒指标。Bounce：controlled_hold_v1，连续0.3秒多区域真实接触受控保持；成功立即结束，不证明更长时间保持。两任务成功率不直接横向比较。

|任务|球|成功/100|触手/100|曾稳定保持/100（Roll）|失败但曾触手|
|---|---|---:|---:|---:|---:|
|roll|original|69|97|21|28|
|roll|tennis_ball|73|97|23|24|
|roll|baseball|85|98|27|13|
|bounce|original|61|99|—|38|
|bounce|tennis_ball|62|99|—|37|
|bounce|baseball|74|100|—|26|

## 诊断

- roll original：终止原因 {'timeout': 75, 'ball_on_floor': 16, 'out_of_bounds': 5, 'catch_success': 2, 'ball_left': 2}；未触手3局；失败但曾触手28局。
  最终区域保持71局，其中正式成功且曾稳定保持21局。失败末态诊断{'below_palm': 26, 'outside_lateral_region': 2, 'other': 3}。
- roll tennis_ball：终止原因 {'timeout': 79, 'ball_on_floor': 12, 'catch_success': 2, 'out_of_bounds': 5, 'ball_left': 2}；未触手3局；失败但曾触手24局。
  最终区域保持76局，其中正式成功且曾稳定保持23局。失败末态诊断{'below_palm': 21, 'outside_lateral_region': 2, 'other': 3, 'above_hand_region': 1}。
- roll baseball：终止原因 {'timeout': 83, 'ball_on_floor': 9, 'catch_success': 4, 'out_of_bounds': 4}；未触手2局；失败但曾触手13局。
  最终区域保持88局，其中正式成功且曾稳定保持27局。失败末态诊断{'below_palm': 11, 'other': 4}。
- bounce original：终止原因 {'controlled_hold': 61, 'timeout': 26, 'out_of_bounds': 10, 'base_collision': 3}；未触手1局；失败但曾触手38局。
  失败中曾满足过有效保持14局；成功末态相对速度中位数0.0156m/s；成功末态拇指参与22局。末控制步最大关节越限0.036022rad，仅末步统计，不代表全程峰值。
- bounce tennis_ball：终止原因 {'controlled_hold': 62, 'timeout': 31, 'out_of_bounds': 7}；未触手1局；失败但曾触手37局。
  失败中曾满足过有效保持15局；成功末态相对速度中位数0.0111m/s；成功末态拇指参与28局。末控制步最大关节越限0.025587rad，仅末步统计，不代表全程峰值。
- bounce baseball：终止原因 {'controlled_hold': 74, 'timeout': 15, 'out_of_bounds': 11}；未触手0局；失败但曾触手26局。
  失败中曾满足过有效保持11局；成功末态相对速度中位数0.0271m/s；成功末态拇指参与48局。末控制步最大关节越限0.022613rad，仅末步统计，不代表全程峰值。

## 解读与限制

Roll主要口径不要求低速度或最短保持时间，不能将区域保持成功等同于稳定抓握。优先检查接触后速度匹配、包球与保持，再考虑改奖励。
Bounce失败可区分完全未触手与触手后无法持续保持。当前成功终止在0.3秒；长期保持及搬运鲁棒性未测。没有hand_joint_limit终止也不能单独证明所有手指姿态正常。
每种球100回合，比例存在抽样误差，95% Wilson区间见summary_all.json。训练日志含随机探索及不同统计窗口，不能当作独立测试结果。
完整逐回合数据、物理参数、模型/源码哈希见各子目录provenance.json、episodes.jsonl、source_hashes.json。

## 手部轨迹补查

每种任务/球另查种子0–9共60局，逐回合success/reward/final与主测试完全一致，确认新增记录未改变轨迹。

- bounce_baseball：全程最大实际关节越限0.0280rad；普通指尖相对掌面最小高度-0.0118m。
  接触采样中不满足项（可重叠，含正常初始碰撞阶段，不等同于失败原因占比）：{'outside_capture': 0.04979442667884879, 'few_contact_groups': 0.31155778894472363, 'small_contact_span': 0.3051621745089082, 'fast_relative_speed': 0.21836455002284147, 'joint_violation': 0.0}。
- bounce_original：全程最大实际关节越限0.0298rad；普通指尖相对掌面最小高度-0.0139m。
  接触采样中不满足项（可重叠，含正常初始碰撞阶段，不等同于失败原因占比）：{'outside_capture': 0.24686759227903826, 'few_contact_groups': 0.2041991195394514, 'small_contact_span': 0.24517439891635626, 'fast_relative_speed': 0.21131053166271588, 'joint_violation': 0.0}。
- bounce_tennis_ball：全程最大实际关节越限0.0288rad；普通指尖相对掌面最小高度-0.0139m。
  接触采样中不满足项（可重叠，含正常初始碰撞阶段，不等同于失败原因占比）：{'outside_capture': 0.015324165029469547, 'few_contact_groups': 0.2907662082514735, 'small_contact_span': 0.287229862475442, 'fast_relative_speed': 0.2306483300589391, 'joint_violation': 0.0}。
- roll_baseball：全程最大实际关节越限0.2273rad；普通指尖相对掌面最小高度0.0049m。
- roll_original：全程最大实际关节越限0.1956rad；普通指尖相对掌面最小高度0.0009m。
- roll_tennis_ball：全程最大实际关节越限0.2123rad；普通指尖相对掌面最小高度0.0051m。

Bounce补查无旧视频数弧度严重越限，但有约1.2–1.4cm指尖短时低于局部掌面；不能据此称手指姿态问题完全消失。Roll约0.20–0.23rad实际越限值得改善控制。

## 与历史Roll252同种子对照

- roll_baseball：旧252为54%，新250为85%；100/100初始状态完全一致。改善41局、退化10局。仅区域保持指标比较，非稳定抓握。
- roll_original：旧252为78%，新250为69%；100/100初始状态完全一致。改善13局、退化22局。仅区域保持指标比较，非稳定抓握。
- roll_tennis_ball：旧252为70%，新250为73%；100/100初始状态完全一致。改善19局、退化16局。仅区域保持指标比较，非稳定抓握。

## 后续建议

Roll优先改善触球后的速度匹配、多指包球和稳定保持，并验证目标限幅/PID约束；保留当前版本做对照。不要仅提高接触或靠近奖励，也不要只提高区域保持数字。Bounce先保留当前约束与成功阈值，训练/评估更新的last与best_success候选；关注手掌接近速度及普通手指/拇指共同包球。若另行修改奖励，可考虑接触阶段速度匹配和持续有效包球，避免强制三指同角。
