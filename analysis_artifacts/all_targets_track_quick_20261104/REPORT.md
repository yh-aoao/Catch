# 2026-11-04 Track 简短评估

使用 01:33:28 目录的 best_reward 模型，每个物体 20 个完整回合，seed=123，单环境同步，无界面，确定性策略。加载各自训练目录 resolved_config.yaml，仅覆盖测试模式、环境数、计算设备、目标与本地资源路径。没有修改奖励、物理或控制逻辑。

Roll 主指标沿用 PPO terminal_metrics 的 roll_eval_success：曾离桌被手拦截，之后发生失败仍可能保留成功。Bounce 为 bounce_touch_v1：碰手且当步无失败。这些都不是 Catch 稳定持球率。

|任务|目标|成功/回合|成功率|
|---|---|---:|---:|
|bounce|baseball|19/20|95%|
|bounce|golf_ball|20/20|100%|
|bounce|original|20/20|100%|
|bounce|racquetball|20/20|100%|
|bounce|softball|20/20|100%|
|bounce|tennis_ball|17/20|85%|
|roll|apple|19/20|95%|
|roll|baseball|17/20|85%|
|roll|lemon|18/20|90%|
|roll|orange|18/20|90%|
|roll|original|16/20|80%|
|roll|tennis_ball|18/20|90%|
|roll|tomato_soup_can|12/20|60%|

20 回合中每回合相当于 5 个百分点，仅用于初步比较，不是正式 benchmark。详细逐回合记录见同目录 JSON/log。

复现入口：catch_it_copy/analysis/run_quick_tracks_20261104.py；评估循环：catch_it_copy/analysis/quick_track_eval_20261104.py。
