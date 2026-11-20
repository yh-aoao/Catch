# Multi-target Catch evaluation

Roll: end-of-episode region retention; stable_hold is ever stable during the episode, not necessarily at the end.
Bounce: controlled_hold_v1 continuous hold. Fixed and DR columns have different test distributions.
Deterministic policy; explicit per-episode launch seeds; DR v1 parameters fixed per instance; dr_episode parameters resampled each episode.
Per-object Wilson intervals and full physics samples are in the JSON results. No failed episodes are filtered.

|Model|Mode|Target|Success|Rate|Roll ever stable|
|---|---|---|---:|---:|---:|
|bounce_best_1119|dr|baseball|107/200|53.5%|—|
|bounce_best_1119|dr|golf_ball|190/200|95.0%|—|
|bounce_best_1119|dr|original|129/200|64.5%|—|
|bounce_best_1119|dr|racquetball|184/200|92.0%|—|
|bounce_best_1119|dr|softball|46/200|23.0%|—|
|bounce_best_1119|dr|tennis_ball|154/200|77.0%|—|

Totals below include completed targets only; check COMPLETE before interpreting a full run.

|Model|Targets completed|Success|Rate|
|---|---:|---:|---:|
|bounce_best_1119|6|810/1200|67.5%|
