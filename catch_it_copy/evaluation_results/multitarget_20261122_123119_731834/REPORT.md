# Multi-target Catch evaluation

Roll: end-of-episode region retention; stable_hold is ever stable during the episode, not necessarily at the end.
Bounce: controlled_hold_v1 continuous hold. Fixed and DR columns have different test distributions.
Deterministic policy; explicit per-episode launch seeds; DR v1 parameters fixed per instance; dr_episode parameters resampled each episode.
Per-object Wilson intervals and full physics samples are in the JSON results. No failed episodes are filtered.

|Model|Mode|Target|Success|Rate|Roll ever stable|
|---|---|---|---:|---:|---:|
|bounce_six_1122|dr|baseball|182/200|91.0%|—|
|bounce_six_1122|dr|golf_ball|113/200|56.5%|—|
|bounce_six_1122|dr|original|172/200|86.0%|—|
|bounce_six_1122|dr|racquetball|86/200|43.0%|—|
|bounce_six_1122|dr|softball|191/200|95.5%|—|
|bounce_six_1122|dr|tennis_ball|178/200|89.0%|—|

Totals below include completed targets only; check COMPLETE before interpreting a full run.

|Model|Targets completed|Success|Rate|
|---|---:|---:|---:|
|bounce_six_1122|6|922/1200|76.8%|
