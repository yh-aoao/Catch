# Creation-time physical DR / revised Roll targets v2

Entry: scripts/train_targets_creation_dr_v2.sh roll|bounce track|catch [overrides].

- physical_dr_v1: shape class fixed per environment, physical parameters sampled once on construction and unchanged on episode resets. Launch position/velocity and existing robot/noise randomization continue per their original reset/control logic. Repeated launch with identical seed can repeat the same physical draws.
- Roll remains seven targets: original, tennis_ball, baseball, orange, apple, tomato_soup_can, small_cylinder. Lemon is excluded from this new catalog only. Older catalogs remain unchanged.
- small_cylinder: synthetic homogeneous solid cylinder, not YCB. Nominal radius .028 m, length .09 m, mass .065 kg; analytical collision and visible cylinder; consistent solid inertia; rolling axis horizontal, no-slip initial spin. speed .8–1.3 m/s, world y2–2.2m. Material generic dry-surface prior, not specimen measurement.
- Bounce six targets unchanged.
- Roll Catch uses roll_palm_version=palm_z_v1. Track logic unchanged. Success conditions unchanged.
- Full DR amplitudes unchanged: size±5%, mass±10%, sliding±15%, Roll rolling±20%, Bounce contact damping ratio±10%. This update does not include the earlier suggested half-range experiment.
- num_envs Roll28 / Bounce24, balanced4 slots per target, caller can lower. This matches Throw's construction-time sampling, while retaining balanced class assignment rather than independent random class counts.
- New outputs: roll_seven_object_track_creation_dr_v2_seed123 / roll_seven_object_catch_creation_dr_v2_seed123 / bounce_six_ball_track_creation_dr_v2_seed123 / bounce_six_ball_catch_creation_dr_v2_seed123.

Train:
```bash
bash scripts/train_targets_creation_dr_v2.sh roll track device_id=0
bash scripts/train_targets_creation_dr_v2.sh bounce track device_id=0
bash scripts/train_targets_creation_dr_v2.sh roll catch checkpoint_tracking=/absolute/path/new_roll_track.pth device_id=0
bash scripts/train_targets_creation_dr_v2.sh bounce catch checkpoint_tracking=/absolute/path/new_bounce_track.pth device_id=0
```

Run one training job at a time on a resource-limited32-core machine. Train script output_name, num_envs etc can be overridden. Old train_all_targets_episode_dr_v1.sh still means per-episode randomization; use the new entry for this experiment.

Validation artifacts: ../analysis_artifacts/creation_dr_v2_20261116.13 targets constructed under creation-time DR, reset three times each; mass/inertia/size identical across resets and launch states differ; finite step rewards. New cylinder crosses table front in40/40 passive launches for one sampled physical instance with varied launch seeds (not a full physical range coverage claim). Short CPU PPO smokes cover Roll/Bounce Track/Catch. Formal training not started.

Backup: ../backups/before_creation_dr_targets_v2_20261116. Palm-reward backup remains ../backups/before_roll_palm_z_v1_20261116.
