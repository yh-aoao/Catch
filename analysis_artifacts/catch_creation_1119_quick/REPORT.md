# Multi-target Catch evaluation

Roll: end-of-episode region retention; stable_hold is ever stable during the episode, not necessarily at the end.
Bounce: controlled_hold_v1 continuous hold. Fixed and DR columns have different test distributions.
Deterministic policy; explicit per-episode launch seeds; DR v1 parameters fixed per instance; dr_episode parameters resampled each episode.
Per-object Wilson intervals and full physics samples are in the JSON results. No failed episodes are filtered.

|Model|Mode|Target|Success|Rate|Roll ever stable|
|---|---|---|---:|---:|---:|
|bounce_best_1119|dr|baseball|12/20|60.0%|—|
|bounce_best_1119|dr|golf_ball|19/20|95.0%|—|
|bounce_best_1119|dr|original|9/20|45.0%|—|
|bounce_best_1119|dr|racquetball|20/20|100.0%|—|
|bounce_best_1119|dr|softball|6/20|30.0%|—|
|bounce_best_1119|dr|tennis_ball|11/20|55.0%|—|
|roll_best_1119|dr|apple|11/20|55.0%|0.0%|
|roll_best_1119|dr|baseball|19/20|95.0%|0.0%|
|roll_best_1119|dr|orange|17/20|85.0%|0.0%|
|roll_best_1119|dr|original|18/20|90.0%|0.0%|
|roll_best_1119|dr|tennis_ball|18/20|90.0%|0.0%|
|roll_best_1119|dr|tomato_soup_can|17/20|85.0%|0.0%|
|roll_best_1119|dr|tuna_fish_can|6/20|30.0%|0.0%|
|roll_last_1119|dr|apple|10/20|50.0%|0.0%|
|roll_last_1119|dr|baseball|15/20|75.0%|0.0%|
|roll_last_1119|dr|orange|15/20|75.0%|0.0%|
|roll_last_1119|dr|original|13/20|65.0%|0.0%|
|roll_last_1119|dr|tennis_ball|16/20|80.0%|0.0%|
|roll_last_1119|dr|tomato_soup_can|7/20|35.0%|0.0%|
|roll_last_1119|dr|tuna_fish_can|8/20|40.0%|0.0%|

Totals below include completed targets only; check COMPLETE before interpreting a full run.

|Model|Targets completed|Success|Rate|
|---|---:|---:|---:|
|bounce_best_1119|6|77/120|64.2%|
|roll_best_1119|7|106/140|75.7%|
|roll_last_1119|7|84/140|60.0%|
