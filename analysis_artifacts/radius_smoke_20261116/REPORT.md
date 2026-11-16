# Multi-target Catch evaluation

Roll: end-of-episode region retention; stable_hold is ever stable during the episode, not necessarily at the end.
Bounce: controlled_hold_v1 continuous hold. Fixed and DR columns have different test distributions.
Deterministic policy; explicit per-episode launch seeds; DR v1 parameters fixed per instance; dr_episode parameters resampled each episode.
Per-object Wilson intervals and full physics samples are in the JSON results. No failed episodes are filtered.

|Model|Mode|Target|Success|Rate|Roll ever stable|
|---|---|---|---:|---:|---:|
|bounce_157|dr_episode|original|0/1|0.0%|—|
|roll_236|dr_episode|original|1/1|100.0%|0.0%|

Totals below include completed targets only; check COMPLETE before interpreting a full run.

|Model|Targets completed|Success|Rate|
|---|---:|---:|---:|
|bounce_157|1|0/1|0.0%|
|roll_236|1|1/1|100.0%|
