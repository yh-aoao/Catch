# Multi-target Catch evaluation

Roll: end-of-episode region retention; stable_hold is ever stable during the episode, not necessarily at the end.
Bounce: controlled_hold_v1 continuous hold. Fixed and DR columns have different test distributions.
Deterministic policy; explicit per-episode launch seeds; DR parameters fixed per environment instance.
Per-object Wilson intervals and full physics samples are in the JSON results. No failed episodes are filtered.

|Model|Mode|Target|Success|Rate|Roll ever stable|
|---|---|---|---:|---:|---:|
|roll_dr_231_61|dr|apple|80/200|40.0%|32.5%|
|roll_dr_231_61|dr|baseball|130/200|65.0%|40.0%|
|roll_dr_231_61|dr|orange|112/200|56.0%|42.0%|
|roll_dr_231_61|dr|original|121/200|60.5%|25.5%|
|roll_dr_231_61|dr|tennis_ball|120/200|60.0%|28.5%|

Totals below include completed targets only; check COMPLETE before interpreting a full run.

|Model|Targets completed|Success|Rate|
|---|---:|---:|---:|
|roll_dr_231_61|5|563/1000|56.3%|
