# Multi-target Catch evaluation

Roll: end-of-episode region retention; stable_hold is ever stable during the episode, not necessarily at the end.
Bounce: controlled_hold_v1 continuous hold. Fixed and DR columns have different test distributions.
Deterministic policy; explicit per-episode launch seeds; DR parameters fixed per environment instance.
Per-object Wilson intervals and full physics samples are in the JSON results. No failed episodes are filtered.

|Model|Mode|Target|Success|Rate|Roll ever stable|
|---|---|---|---:|---:|---:|
|bounce_dr_best_4_84|dr|baseball|123/200|61.5%|—|
|bounce_dr_best_4_84|dr|golf_ball|187/200|93.5%|—|
|bounce_dr_best_4_84|dr|original|116/200|58.0%|—|
|bounce_dr_best_4_84|dr|racquetball|185/200|92.5%|—|
|bounce_dr_best_4_84|dr|softball|164/200|82.0%|—|
|bounce_dr_best_4_84|dr|tennis_ball|140/200|70.0%|—|
|bounce_dr_last|dr|baseball|92/200|46.0%|—|
|bounce_dr_last|dr|golf_ball|178/200|89.0%|—|
|bounce_dr_last|dr|original|88/200|44.0%|—|
|bounce_dr_last|dr|racquetball|177/200|88.5%|—|
|bounce_dr_last|dr|softball|150/200|75.0%|—|
|bounce_dr_last|dr|tennis_ball|124/200|62.0%|—|

Totals below include completed targets only; check COMPLETE before interpreting a full run.

|Model|Targets completed|Success|Rate|
|---|---:|---:|---:|
|bounce_dr_best_4_84|6|915/1200|76.2%|
|bounce_dr_last|6|809/1200|67.4%|
