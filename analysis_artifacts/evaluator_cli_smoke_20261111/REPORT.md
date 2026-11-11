# Multi-target Catch evaluation

Roll: end-of-episode region retention; stable_hold is ever stable during the episode, not necessarily at the end.
Bounce: controlled_hold_v1 continuous hold. Fixed and DR columns have different test distributions.
Deterministic policy; explicit per-episode launch seeds; DR parameters fixed per environment instance.
Per-object Wilson intervals and full physics samples are in the JSON results. No failed episodes are filtered.

|Model|Mode|Target|Success|Rate|Roll ever stable|
|---|---|---|---:|---:|---:|
|bounce_dr_3_96|dr|baseball|3/4|75.0%|—|
|bounce_dr_3_96|dr|golf_ball|3/4|75.0%|—|
|bounce_dr_3_96|dr|original|1/4|25.0%|—|
|bounce_dr_3_96|dr|racquetball|4/4|100.0%|—|
|bounce_dr_3_96|dr|softball|3/4|75.0%|—|
|bounce_dr_3_96|dr|tennis_ball|2/4|50.0%|—|
|bounce_fixed_3_39|fixed|baseball|2/4|50.0%|—|
|bounce_fixed_3_39|fixed|golf_ball|3/4|75.0%|—|
|bounce_fixed_3_39|fixed|original|0/4|0.0%|—|
|bounce_fixed_3_39|fixed|racquetball|1/4|25.0%|—|
|bounce_fixed_3_39|fixed|softball|2/4|50.0%|—|
|bounce_fixed_3_39|fixed|tennis_ball|2/4|50.0%|—|
|roll_dr_relaunch_222_64|dr|apple|3/4|75.0%|0.0%|
|roll_dr_relaunch_222_64|dr|baseball|3/4|75.0%|50.0%|
|roll_dr_relaunch_222_64|dr|lemon|1/4|25.0%|25.0%|
|roll_dr_relaunch_222_64|dr|orange|3/4|75.0%|50.0%|
|roll_dr_relaunch_222_64|dr|original|2/4|50.0%|0.0%|
|roll_dr_relaunch_222_64|dr|tennis_ball|0/4|0.0%|0.0%|
|roll_dr_relaunch_222_64|dr|tomato_soup_can|1/4|25.0%|0.0%|
|roll_fixed_261_94|fixed|apple|4/4|100.0%|0.0%|
|roll_fixed_261_94|fixed|baseball|4/4|100.0%|0.0%|
|roll_fixed_261_94|fixed|lemon|0/4|0.0%|0.0%|
|roll_fixed_261_94|fixed|orange|4/4|100.0%|0.0%|
|roll_fixed_261_94|fixed|original|3/4|75.0%|0.0%|
|roll_fixed_261_94|fixed|tennis_ball|4/4|100.0%|0.0%|
|roll_fixed_261_94|fixed|tomato_soup_can|3/4|75.0%|0.0%|

Totals below include completed targets only; check COMPLETE before interpreting a full run.

|Model|Targets completed|Success|Rate|
|---|---:|---:|---:|
|bounce_dr_3_96|6|16/24|66.7%|
|bounce_fixed_3_39|6|10/24|41.7%|
|roll_dr_relaunch_222_64|7|13/28|46.4%|
|roll_fixed_261_94|7|22/28|78.6%|
