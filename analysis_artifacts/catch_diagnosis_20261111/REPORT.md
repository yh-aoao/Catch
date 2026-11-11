# Roll/Bounce Catch diagnosis, 2026-11-11

Only analysis scripts and evaluation artifacts were added; production rewards, launch defaults, training and physics files are unchanged.

## Controlled diagnostic evaluation

10 physical instances per case; 60 episodes per Bounce case, 40 per Roll case; deterministic policy, explicit launch seeds. Range 1 is current DR, 0.5 scales each relative deviation by half, 0 uses nominal physics with the same launch sampling. All failures are retained.

|Case|Successes|Rate|Wilson 95% CI|
|---|---:|---:|---|
|bounce_original_range1.0|38/60|63.3%|50.7%–74.4%|
|bounce_original_range0.5|37/60|61.7%|49.0%–72.9%|
|bounce_original_range0.0|35/60|58.3%|45.7%–69.9%|
|bounce_tennis_ball_range1.0|43/60|71.7%|59.2%–81.5%|
|bounce_tennis_ball_range0.5|43/60|71.7%|59.2%–81.5%|
|bounce_tennis_ball_range0.0|46/60|76.7%|64.6%–85.6%|
|roll_apple_range1.0|12/40|30.0%|18.1%–45.4%|
|roll_apple_range0.5|14/40|35.0%|22.1%–50.5%|
|roll_apple_range0.0|14/40|35.0%|22.1%–50.5%|
|roll_lemon_range1.0|11/40|27.5%|16.1%–42.8%|
|roll_lemon_range0.5|15/40|37.5%|24.2%–53.0%|
|roll_lemon_range0.0|16/40|40.0%|26.3%–55.4%|
|bounce_original_spin0|33/60|55.0%|42.5%–66.9%|
|bounce_original_nearer|42/60|70.0%|57.5%–80.1%|
|bounce_original_threeball|55/60|91.7%|81.9%–96.4%|

## Interpretation and limits

- Same-model inference ablations diagnose sensitivity; they do not prove that retraining with a smaller range will improve performance. Small samples and shared physical instances limit precision.
- Same launch seeds/states are verified. Subsequent noise is not strictly paired because historical synchronous environments share a global NumPy RNG and episode lengths/resets diverge.
- Original six-ball policy: inspect contact_tick_gate_fractions and never_valid_hold in summary.json. Contact alone is not a grasp. Most existing 200-episode failures had hand contact, but could not maintain capture/contact geometry/low relative speed together.
- Original legacy route uses a different launch sampler from new benchmark balls: larger lateral/heading spread and forward-speed-dependent spin. New balls have material-aware rebound-apex launches and zero angular velocity. This is a distribution difference, not by itself a proven causal explanation.
- spin0 removes X/Y speed-dependent spin only; the original Z spin sample [-3,3] rad/s remains. This intervention is diagnostic and is not a changed default.
- nearer changes only the initial Y-distance range from [1.6,2.0] to [1.6,1.8] m. It is not a benchmark improvement unless the changed launch distribution is explicitly reported.
- The old three-ball and current six-ball checkpoints are evaluated under identical current original-ball physics and launch states. Differences establish checkpoint capability differences, but cannot separate all historical training-recipe causes without a matched training experiment.

## Physical randomization versus Throw

Current relative ranges: dimensions/radius ±5%; mass ±10%; floor/table sliding friction ±15%; Roll rolling length ±20%; Bounce contact damping ratio ±10%. Inertia follows mass times size squared. The damping ratio is not a restitution coefficient.
Throw sphere radius is 0.035–0.045 m (±12.5% about 0.04 m); free-joint damping is 0.005–0.02, unlike contact damping. Throw mass configuration says 0.035–0.075 kg but its implementation calls uniform(low,low), so actual sampled mass is fixed at 0.035 kg. It does not randomize the same surface contact parameters as this DR version.
All these physics draws occur at environment creation, not every episode. Default multi-target scripts use 28 Roll / 24 Bounce environments, ordinarily four physical profiles per target. Actual remote run overrides require its saved training config/provenance to verify.

## Recommended next experiments

1. Preserve current model/physics/reward snapshots; use validation success, not training best reward alone, to select checkpoints. Current Bounce best is better than last; last did not improve with more steps.
2. Fine-tune a strong Catch policy into the multi-target task rather than necessarily reinitializing Catch from Track. Preserve old-target skills with old-policy distillation or balanced nominal/DR coverage. Use smaller joint-update learning rates and compare Track learning-rate reduction against the current equal-rate recipe.
3. Increase physics-profile coverage: per-reset resampling where safe, or a larger prevalidated profile pool. Include held-out profile/launch seeds for validation.
4. Use half-range as an optional training curriculum, then expand to current ranges; keep final evaluation distribution unchanged and report both nominal and randomized performance. Do not declare smaller ranges necessary based on this small inference ablation.
5. Improve capture-to-hold learning: review existing capture, contact coverage, relative-speed and holding-progress shaping before adding more rewards. Keep real-contact/off-ground success requirements. Consider observable size/aspect features or history/recurrent inputs; present policy lacks explicit object size, material and angular velocity.
6. Keep Roll held-at-end and stable-hold metrics separate. Fixed Roll 261.94 has 62.9% region-retention success in the existing test but 0% ever-stable hold; a higher primary rate alone does not demonstrate better grip quality.

Raw per-episode diagnostics, parameter draws, commands/scripts and source/checkpoint hashes are retained in this directory.
