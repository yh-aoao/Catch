# Multi-target Catch evaluation

Roll: end-of-episode region retention; stable_hold is ever stable during the episode, not necessarily at the end.
Bounce: controlled_hold_v1 continuous hold. Fixed and DR columns have different test distributions.
Deterministic policy; explicit per-episode launch seeds; DR v1 parameters fixed per instance; dr_episode parameters resampled each episode.
Per-object Wilson intervals and full physics samples are in the JSON results. No failed episodes are filtered.

|Model|Mode|Target|Success|Rate|Roll ever stable|
|---|---|---|---:|---:|---:|
|bounce_best_1_57|dr_episode|baseball|2/20|10.0%|—|
|bounce_best_1_57|dr_episode|golf_ball|16/20|80.0%|—|
|bounce_best_1_57|dr_episode|original|8/20|40.0%|—|
|bounce_best_1_57|dr_episode|racquetball|16/20|80.0%|—|
|bounce_best_1_57|dr_episode|softball|11/20|55.0%|—|
|bounce_best_1_57|dr_episode|tennis_ball|10/20|50.0%|—|
|bounce_last|dr_episode|baseball|6/20|30.0%|—|
|bounce_last|dr_episode|golf_ball|14/20|70.0%|—|
|bounce_last|dr_episode|original|5/20|25.0%|—|
|bounce_last|dr_episode|racquetball|15/20|75.0%|—|
|bounce_last|dr_episode|softball|14/20|70.0%|—|
|bounce_last|dr_episode|tennis_ball|9/20|45.0%|—|
|roll_best_236_00|dr_episode|apple|7/20|35.0%|0.0%|
|roll_best_236_00|dr_episode|baseball|11/20|55.0%|20.0%|
|roll_best_236_00|dr_episode|lemon|4/20|20.0%|15.0%|
|roll_best_236_00|dr_episode|orange|9/20|45.0%|5.0%|
|roll_best_236_00|dr_episode|original|12/20|60.0%|5.0%|
|roll_best_236_00|dr_episode|tennis_ball|9/20|45.0%|0.0%|
|roll_best_236_00|dr_episode|tomato_soup_can|4/20|20.0%|5.0%|

Totals below include completed targets only; check COMPLETE before interpreting a full run.

|Model|Targets completed|Success|Rate|
|---|---:|---:|---:|
|bounce_best_1_57|6|63/120|52.5%|
|bounce_last|6|63/120|52.5%|
|roll_best_236_00|7|56/140|40.0%|
