# Multi-target Catch evaluation

Roll: end-of-episode region retention; stable_hold is ever stable during the episode, not necessarily at the end.
Bounce: controlled_hold_v1 continuous hold. Fixed and DR columns have different test distributions.
Deterministic policy; explicit per-episode launch seeds; DR parameters fixed per environment instance.
Per-object Wilson intervals and full physics samples are in the JSON results. No failed episodes are filtered.

|Model|Mode|Target|Success|Rate|Roll ever stable|
|---|---|---|---:|---:|---:|
|roll_dr_229_39|dr|apple|48/200|24.0%|18.0%|
|roll_dr_229_39|dr|baseball|112/200|56.0%|34.0%|
|roll_dr_229_39|dr|lemon|79/200|39.5%|16.5%|
|roll_dr_229_39|dr|orange|98/200|49.0%|28.5%|
|roll_dr_229_39|dr|original|94/200|47.0%|21.5%|
|roll_dr_229_39|dr|tennis_ball|114/200|57.0%|24.5%|
|roll_dr_229_39|dr|tomato_soup_can|85/200|42.5%|36.0%|
|roll_dr_relaunch_222_64|dr|apple|72/200|36.0%|26.5%|
|roll_dr_relaunch_222_64|dr|baseball|121/200|60.5%|28.0%|
|roll_dr_relaunch_222_64|dr|lemon|60/200|30.0%|17.5%|
|roll_dr_relaunch_222_64|dr|orange|102/200|51.0%|28.0%|
|roll_dr_relaunch_222_64|dr|original|118/200|59.0%|14.0%|
|roll_dr_relaunch_222_64|dr|tennis_ball|116/200|58.0%|18.0%|
|roll_dr_relaunch_222_64|dr|tomato_soup_can|80/200|40.0%|26.5%|
|roll_fixed_261_94|fixed|apple|102/200|51.0%|0.0%|
|roll_fixed_261_94|fixed|baseball|171/200|85.5%|0.0%|
|roll_fixed_261_94|fixed|lemon|37/200|18.5%|0.0%|
|roll_fixed_261_94|fixed|orange|168/200|84.0%|0.0%|
|roll_fixed_261_94|fixed|original|156/200|78.0%|0.0%|
|roll_fixed_261_94|fixed|tennis_ball|165/200|82.5%|0.0%|
|roll_fixed_261_94|fixed|tomato_soup_can|82/200|41.0%|0.0%|

Totals below include completed targets only; check COMPLETE before interpreting a full run.

|Model|Targets completed|Success|Rate|
|---|---:|---:|---:|
|roll_dr_229_39|7|630/1400|45.0%|
|roll_dr_relaunch_222_64|7|669/1400|47.8%|
|roll_fixed_261_94|7|881/1400|62.9%|
