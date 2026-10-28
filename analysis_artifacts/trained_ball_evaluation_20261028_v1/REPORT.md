# 新模型逐球评估（2026-10-28）

每个条件 100 回合，共 600 回合，seed=0–99。使用新模型 best checkpoint，CPU 无 Viewer，2.5 s 回合上限、40 ms 策略周期。
保留训练物理/材质/发射配置，Roll 初速 0.8–1.3 m/s；固定测试每种球，不按训练比例混抽。两份材质文件 SHA256 与训练记录一致。

|任务|球类|现有主判据成功|95% Wilson 区间|环境原始 success 标志|手接触回合|连续手接触≥0.3s|主判据成功且连续接触≥0.3s|
|---|---|---:|---:|---:|---:|---:|---:|
|roll_track|原始参考球|65/100 (65%)|55.3–73.6%|36/100|65/100|0/100|0/100|
|roll_track|网球|53/100 (53%)|43.3–62.5%|21/100|53/100|0/100|0/100|
|roll_track|棒球|63/100 (63%)|53.2–71.8%|30/100|63/100|0/100|0/100|
|bounce_catch|原始参考球|24/100 (24%)|16.7–33.2%|24/100|97/100|23/100|20/100|
|bounce_catch|网球|34/100 (34%)|25.5–43.7%|34/100|97/100|36/100|28/100|
|bounce_catch|棒球|8/100 (8%)|4.1–15.0%|8/100|82/100|13/100|6/100|

## 判据和限制

- Roll Track 的主判据是 roll_eval_success：球完整越过桌前沿、无桌/地支撑时，曾被手接触拦截。不是完整抓取，也不要求之后继续持稳。
- Roll 环境原始 success 更侧重掌心触球且无失败，与主判据不相同；两列同时保留。
- Bounce 主判据是 bounce_catch_proximity_v2：到达回合时限、无失败终止、grasping 阶段、ee_distance<0.25 m。不能称为严格持球成功率。
- 持续接触按 2 ms 控制子步检测，≥0.3 s 只是补充证据，并非力闭合/末端稳持判据。Roll Track 触球即可能结束，持续接触不适合评价其策略。
- 权重严格加载，未允许尺寸不匹配的部分加载；调用原 PPO 的 obs2tensor/model_act(inference=True)/action2dict，未训练或更改权重。
- 每个子目录保存逐回合 JSONL、模型和配置/材质 SHA256、实际物理参数、终止原因、首末状态和源码 SHA256。

物理工厂与材质文件 SHA256 与训练记录一致；共享 DcmmVecEnv.py 的整文件 SHA256 不同，未找到对应历史源码备份，故这些是当前本地代码下的评估，不声称逐字复现训练机历史共享模块。
原训练配置中的 mixed 字段保留在 PPO 配置快照中；实际工厂明确按当前球固定测试，不混合采样。

## 终止原因

- roll_track / 原始参考球：`{"collision_or_failure": 61, "track_touch": 36, "timeout": 3}`
- roll_track / 网球：`{"collision_or_failure": 75, "track_touch": 21, "timeout": 4}`
- roll_track / 棒球：`{"track_touch": 30, "collision_or_failure": 69, "timeout": 1}`
- bounce_catch / 原始参考球：`{"ball_left": 71, "catch_success": 24, "failure_unspecified": 5}`
- bounce_catch / 网球：`{"catch_success": 34, "ball_left": 60, "failure_unspecified": 6}`
- bounce_catch / 棒球：`{"ball_left": 90, "catch_success": 8, "failure_unspecified": 2}`

## 模型

- roll_track: `/home/isee/Catch/roll_three_ball_tracking_legacy_reward_v1_seed123/2026-10-28/11:36:06/nn/roll_track_best_reward_23.86.pth`
  SHA256: `d188548fc27be1ea98cd98d607d4fd7304d416db0772a1e2241e977b7df65687`
- bounce_catch: `/home/isee/Catch/bounce_three_ball_catching_v1_seed123/2026-10-28/11:36:32/nn/bounce_catch_best_reward_288.46.pth`
  SHA256: `5d580d59cf4d3cdace14457351c35434fa44aaf82a08babfe5a2b8968bb2b63d`


## 后续源码核验补充（2026-10-28）

已从 Git 提交 `278ca2b4d92489784edee3a55c1fa820181aa525` 找到与训练记录整文件 SHA256 完全相同的 `DcmmVecEnv.py`。该历史文件中的整个 `BounceEnv` 类与当前版本文本完全相同，包括奖励、手部观测、阶段切换与仿真控制方法；`physical_v2` 工厂文件也与训练记录哈希相同。此前“尚未找到历史源码”的限制已针对 BounceEnv 解决。此结论不扩展为所有依赖均已一致。详见[核验报告](../bounce_reward_training_source_20261028/REPORT.md)。
