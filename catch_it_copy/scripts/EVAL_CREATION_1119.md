# 11-19 Catch evaluation

Use the dcmm environment, working directory catch_it_copy.

```bash
python scripts/evaluate_multitarget_catch.py \
 --models-file scripts/eval_creation_dr_1119.json \
 --catalog ycb_objects/calibration_data/object_benchmark_creation_v2.json \
 --episodes 200 --instances 4 --jobs 2 --seed 41000
```

`--episodes` counts completed episodes PER MODEL AND TARGET, not the overall total. Four models,7 Roll targets and6 Bounce targets mean26 cases:20 gives520 episodes;200 gives5200. Each case uses instances synchronous environments with distinct creation-time sampled physical draws; each instance runs episodes/instances episodes with deterministic per-episode launch seeds. Episode count must be divisible by instances. This is NOT per-episode physical randomization.

Use `--only roll_best_1119`, `--only roll_last_1119`, `--only bounce_best_1119` or `--only bounce_last_1119` for one model; multiple names allowed. `--objects original tennis_ball` restricts targets. Roll catalog excludes lemon and includes tuna_fish_can. Roll models specify palm_z_v1; historical manifests default to legacy. Bounce uses controlled_hold_v1 via production entry.

Physics defaults to full parameter range used by creation_dr_v2, not nominal-only, and not half-range. Four physical draws per object are a quick screen, not broad range coverage. Increase --instances (e.g.20 for200 episodes) to broaden physical samples; jobs controls concurrent worker processes. Current sync backend runs env steps sequentially inside each worker. Increasing instances increases memory, not worker-process count.

Default output evaluation_results/multitarget_TIMESTAMP/ contains REPORT.md, summary.csv, model_totals.json, full per-case physics+episodes. COMPLETE means all cases finished. No training, no changed success thresholds.

Roll primary: final region retention with failure override; strict ever-stable is separately reported. Bounce primary: continuous controlled hold for0.3s. Do not compare Roll primary and Bounce primary as identical grasp-quality metrics. Best checkpoint filename reward is not success rate.
