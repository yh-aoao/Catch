# Multi-target Catch evaluation

Roll: end-of-episode region retention; stable_hold is ever stable during the episode, not necessarily at the end.
Bounce: controlled_hold_v1 continuous hold. Fixed and DR columns have different test distributions.
Deterministic policy; explicit per-episode launch seeds; DR v1 parameters fixed per instance; dr_episode parameters resampled each episode.
Per-object Wilson intervals and full physics samples are in the JSON results. No failed episodes are filtered.

|Model|Mode|Target|Success|Rate|Roll ever stable|
|---|---|---|---:|---:|---:|
|roll_creation_dr_v3|dr|apple|13/20|65.0%|40.0%|
|roll_creation_dr_v3|dr|baseball|17/20|85.0%|65.0%|
|roll_creation_dr_v3|dr|coke_can|18/20|90.0%|85.0%|
|roll_creation_dr_v3|dr|orange|16/20|80.0%|70.0%|
|roll_creation_dr_v3|dr|original|18/20|90.0%|60.0%|
|roll_creation_dr_v3|dr|tennis_ball|19/20|95.0%|65.0%|
|roll_creation_dr_v3|dr|tomato_soup_can|19/20|95.0%|80.0%|

Totals below include completed targets only; check COMPLETE before interpreting a full run.

|Model|Targets completed|Success|Rate|
|---|---:|---:|---:|
|roll_creation_dr_v3|7|120/140|85.7%|
