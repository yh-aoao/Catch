# Multi-target Catch evaluation

Roll: end-of-episode region retention; stable_hold is ever stable during the episode, not necessarily at the end.
Bounce: controlled_hold_v1 continuous hold. Fixed and DR columns have different test distributions.
Deterministic policy; explicit per-episode launch seeds; DR v1 parameters fixed per instance; dr_episode parameters resampled each episode.
Per-object Wilson intervals and full physics samples are in the JSON results. No failed episodes are filtered.

|Model|Mode|Target|Success|Rate|Roll ever stable|
|---|---|---|---:|---:|---:|
|bounce_last_1119|dr|baseball|5/20|25.0%|—|
|bounce_last_1119|dr|golf_ball|16/20|80.0%|—|
|bounce_last_1119|dr|original|12/20|60.0%|—|
|bounce_last_1119|dr|racquetball|19/20|95.0%|—|
|bounce_last_1119|dr|softball|3/20|15.0%|—|
|bounce_last_1119|dr|tennis_ball|12/20|60.0%|—|

Totals below include completed targets only; check COMPLETE before interpreting a full run.

|Model|Targets completed|Success|Rate|
|---|---:|---:|---:|
|bounce_last_1119|6|67/120|55.8%|
