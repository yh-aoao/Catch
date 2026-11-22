# Multi-target Catch evaluation

Roll: end-of-episode region retention; stable_hold is ever stable during the episode, not necessarily at the end.
Bounce: controlled_hold_v1 continuous hold. Fixed and DR columns have different test distributions.
Deterministic policy; explicit per-episode launch seeds; DR v1 parameters fixed per instance; dr_episode parameters resampled each episode.
Per-object Wilson intervals and full physics samples are in the JSON results. No failed episodes are filtered.

|Model|Mode|Target|Success|Rate|Roll ever stable|
|---|---|---|---:|---:|---:|
|roll_creation_dr_v3|dr|apple|71/100|71.0%|53.0%|
|roll_creation_dr_v3|dr|baseball|93/100|93.0%|63.0%|
|roll_creation_dr_v3|dr|coke_can|90/100|90.0%|74.0%|
|roll_creation_dr_v3|dr|orange|89/100|89.0%|65.0%|
|roll_creation_dr_v3|dr|original|90/100|90.0%|41.0%|
|roll_creation_dr_v3|dr|tennis_ball|93/100|93.0%|58.0%|
|roll_creation_dr_v3|dr|tomato_soup_can|89/100|89.0%|70.0%|

Totals below include completed targets only; check COMPLETE before interpreting a full run.

|Model|Targets completed|Success|Rate|
|---|---:|---:|---:|
|roll_creation_dr_v3|7|615/700|87.9%|
