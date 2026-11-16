# 2026-11-16 Catch 小样本评估

每目标20回合，4个同步环境各5回合；每回合重新采样尺寸、质量与接触参数（physical_dr_episode_v1）。每目标核验20组不同物理参数。固定确定性策略，发射基种子41000；保留所有失败回合。

|模型|目标|成功次数|成功率|Roll稳定持球率|失败原因|
|---|---|---:|---:|---:|---|
|roll_best_236_00|original|12/20|60.0%|5.0%|{'ball_left': 1, 'ball_on_floor': 6, 'out_of_bounds': 1}|
|roll_best_236_00|tennis_ball|9/20|45.0%|0.0%|{'ball_left': 2, 'ball_on_floor': 6, 'out_of_bounds': 2, 'timeout': 1}|
|roll_best_236_00|baseball|11/20|55.0%|20.0%|{'ball_on_floor': 4, 'base_collision': 1, 'out_of_bounds': 2, 'timeout': 2}|
|roll_best_236_00|orange|9/20|45.0%|5.0%|{'ball_left': 1, 'ball_on_floor': 8, 'timeout': 2}|
|roll_best_236_00|apple|7/20|35.0%|0.0%|{'ball_left': 2, 'ball_on_floor': 10, 'timeout': 1}|
|roll_best_236_00|lemon|4/20|20.0%|15.0%|{'ball_left': 4, 'ball_on_floor': 11, 'timeout': 1}|
|roll_best_236_00|tomato_soup_can|4/20|20.0%|5.0%|{'ball_left': 3, 'ball_on_floor': 11, 'out_of_bounds': 2}|
|bounce_best_1_57|original|8/20|40.0%|—|{'base_collision': 1, 'out_of_bounds': 6, 'timeout': 5}|
|bounce_best_1_57|tennis_ball|10/20|50.0%|—|{'base_collision': 3, 'out_of_bounds': 3, 'timeout': 4}|
|bounce_best_1_57|baseball|2/20|10.0%|—|{'out_of_bounds': 11, 'timeout': 7}|
|bounce_best_1_57|racquetball|16/20|80.0%|—|{'base_collision': 2, 'out_of_bounds': 1, 'timeout': 1}|
|bounce_best_1_57|golf_ball|16/20|80.0%|—|{'base_collision': 3, 'timeout': 1}|
|bounce_best_1_57|softball|11/20|55.0%|—|{'out_of_bounds': 5, 'timeout': 4}|
|bounce_last|original|5/20|25.0%|—|{'base_collision': 3, 'out_of_bounds': 5, 'timeout': 7}|
|bounce_last|tennis_ball|9/20|45.0%|—|{'base_collision': 3, 'out_of_bounds': 4, 'timeout': 4}|
|bounce_last|baseball|6/20|30.0%|—|{'out_of_bounds': 6, 'timeout': 8}|
|bounce_last|racquetball|15/20|75.0%|—|{'base_collision': 2, 'out_of_bounds': 1, 'timeout': 2}|
|bounce_last|golf_ball|14/20|70.0%|—|{'base_collision': 3, 'out_of_bounds': 2, 'timeout': 1}|
|bounce_last|softball|14/20|70.0%|—|{'out_of_bounds': 5, 'timeout': 1}|

|模型|成功次数|总体成功率|Roll稳定持球率|
|---|---:|---:|---:|
|roll_best_236_00|56/140|40.0%|7.1%|
|bounce_best_1_57|63/120|52.5%|—|
|bounce_last|63/120|52.5%|—|

Roll主指标为回合结束时region_retention_v5：发生真实手接触并离桌、无外部支撑，物体最后仍在手部保留区域；不要求低相对速度。稳定持球是回合中曾连续满足真实接触、低相对速度等条件约0.3秒，不保证最后仍稳定。
Bounce采用controlled_hold_v1：离地、无外部支撑、多部位接触及低相对速度等条件连续成立约0.3秒。Roll与Bounce的成功判定不同，不宜直接横向比较。
Bounce best和last已核验所有对应回合的初始位置、速度、物理参数完全一致；同步环境沿用全局随机数，后续噪声不保证严格逐步配对。
每目标20回合，单回合对应5个百分点；Wilson区间见validated_summary.json，不能把小差异认定为确定提升。
现有历史固定参数/每环境采样结果与本次逐回合采样的分布不同，不应仅按数字直接判断训练方案优劣。
原始配置、checkpoint及源文件哈希见results/run_spec.json；每回合初始状态、终止原因及物理参数见results/模型名/目标名/episodes.jsonl。未启动训练。
