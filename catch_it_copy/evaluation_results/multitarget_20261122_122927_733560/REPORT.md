# Multi-target Catch evaluation

Roll: end-of-episode region retention; stable_hold is ever stable during the episode, not necessarily at the end.
Bounce: controlled_hold_v1 continuous hold. Fixed and DR columns have different test distributions.
Deterministic policy; explicit per-episode launch seeds; DR v1 parameters fixed per instance; dr_episode parameters resampled each episode.
Per-object Wilson intervals and full physics samples are in the JSON results. No failed episodes are filtered.

|Model|Mode|Target|Success|Rate|Roll ever stable|
|---|---|---|---:|---:|---:|
|roll_seven_1122|dr|apple|155/200|77.5%|63.0%|
|roll_seven_1122|dr|baseball|181/200|90.5%|64.0%|
|roll_seven_1122|dr|coke_can|170/200|85.0%|64.0%|
|roll_seven_1122|dr|orange|185/200|92.5%|73.5%|
|roll_seven_1122|dr|original|177/200|88.5%|44.5%|
|roll_seven_1122|dr|tennis_ball|183/200|91.5%|58.0%|
|roll_seven_1122|dr|tomato_soup_can|181/200|90.5%|73.5%|

Totals below include completed targets only; check COMPLETE before interpreting a full run.

|Model|Targets completed|Success|Rate|
|---|---:|---:|---:|
|roll_seven_1122|7|1232/1400|88.0%|
