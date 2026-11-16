# Roll Catch palm_z_v1

Switch: roll_palm_version=palm_z_v1; default legacy preserves prior reward calculations. Backup: ../backups/before_roll_palm_z_v1_20261116 (includes old config/training/environment/helpers).

Changes are in the Catch environment only:
- Palm normal is link6 +Z. Estimate palm face reference from palm mesh mean X/Y and 90th-percentile Z in link6 frame (geometric prior, not a measured contact plane).
- Capture point is this reference plus +Z times target support distance. Spheres use radius; benchmark fruit/cylinder use projected collision support points in current object orientation, including per-episode size scaling.
- Palm-up shaping uses Z-up alignment instead of Y-up. Existing weights are preserved; orientation reward fades to zero over 0.3 s after the grip state's first valid hand contact with no table/floor support. Before contact it remains active. Contact-loss/grip-state reset can re-enable guidance.
- Finger entry/posture gate uses Z as depth and X/Y as palm plane. Existing post-contact bypass is preserved.
- No changes to success criteria, policy observation/action dimensions, target catalog, launcher, mass/material ranges or PPO settings. Default seven targets still includes lemon. Full per-episode physical randomization remains enabled.

Track uses a separate environment and is unchanged by this switch. Existing compatible Track is suitable for starting new Catch training; retraining Track with this flag does not test the Catch reward change. Start Catch afresh from Track rather than silently evaluating existing Catch under changed reward geometry.

Command:

```bash
conda activate dcmm
cd /home/isee/Catch/catch_it_copy
bash scripts/train_roll_palm_z_v1.sh catch \
 checkpoint_tracking=/home/isee/Catch/11-12/roll_track_best_reward_24.52.pth \
 device_id=0
```

Defaults num_envs=28, seed123, output_name=roll_seven_object_catch_palm_z_v1_seed123; max_agent_steps=1e9 is merely an upper limit. Override num_envs=21 or14 for lower CPU/RAM pressure. Increasing environment count is not part of this experiment.

Validation: all seven targets constructed/reset under per-episode DR and stepped five policy ticks with finite rewards/capture points; CPU synchronous PPO Catch warm initialization from Track and actual updates completed with max_agent_steps=16, horizon8. Logs: ../analysis_artifacts/roll_palm_z_v1_20261116. This verifies execution, not success-rate improvement. Formal training not started.
