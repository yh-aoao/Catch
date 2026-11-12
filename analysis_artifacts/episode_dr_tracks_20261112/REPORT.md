# 2026-11-12 Track 小样本评估

模型：11-12/roll_track_best_reward_24.52.pth；11-12/bounce_track_best_reward_19.56.pth。
每目标20回合，4个同步环境每槽5回合；每个回合重新采样物理参数（physical_dr_episode_v1），每个目标验证20组不同参数。CPU确定性策略，显式逐回合发射种子（base=41000），无失败过滤。

|任务|目标|成功次数|成功率|失败原因|
|---|---|---:|---:|---|
|roll|original|19/20|95.0%|{'collision_or_failure': 1}|
|roll|tennis_ball|19/20|95.0%|{'collision_or_failure': 1}|
|roll|baseball|19/20|95.0%|{'collision_or_failure': 1}|
|roll|orange|19/20|95.0%|{'collision_or_failure': 1}|
|roll|apple|19/20|95.0%|{'collision_or_failure': 1}|
|roll|lemon|20/20|100.0%|{}|
|roll|tomato_soup_can|20/20|100.0%|{}|
|bounce|original|18/20|90.0%|{'early_contact': 2}|
|bounce|tennis_ball|18/20|90.0%|{'early_contact': 2}|
|bounce|baseball|20/20|100.0%|{}|
|bounce|racquetball|20/20|100.0%|{}|
|bounce|golf_ball|20/20|100.0%|{}|
|bounce|softball|19/20|95.0%|{'base_collision': 1}|

|任务|合计|成功率|
|---|---:|---:|
|roll|135/140|96.4%|
|bounce|115/120|95.8%|

Roll使用实际环境roll_eval_success：物体离桌且无桌面/地面支撑时与手发生接触即记录拦截。它是曾经成功拦截的指标，之后落地或终止不抹去该Track记录。原始终止原因可能仍是collision_or_failure，不能把成功率理解为最终抓稳率。
Bounce使用实际环境bounce_touch_v1：手接触且同一步没有失败；不要求持续持球。
因此两项结果都是Track指标，后续Catch成功率必须独立测试。每目标20回合，单回合对应5个百分点，不适合据此判断很小的模型差异。
物理范围和发射条件采用当前完整目标配置；旧11-6测试使用每环境固定物理参数，不能把其数值变化直接解释为模型提升。
源文件、checkpoint哈希与有效配置保存在run_spec.json和roll.yaml/bounce.yaml；每回合初始状态和物理参数保存在各目标episodes.jsonl。
