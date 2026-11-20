# Held-out object evaluation v1

Training catalogs unchanged. Independent catalog: ycb_objects/calibration_data/object_benchmark_unseen_v1.json. Models: scripts/eval_unseen_1119.json (Roll376.90,palm_z_v1; Bounce2.11,controlled_hold_v1). "Unseen" means object identity absent from the current seven/six-target training catalog; these are still known shape families and this is not a claim of unseen-category generalization.

Roll peach: YCB scanned visual and convex collision, imported asset mass, homogeneous mesh inertia estimate. Surface priors follow existing fruit: sliding0.4, rolling length2e-5m. y1.9–2.1m, speed1.4–1.7m/s. YCB fruit sample, not a soft fresh-fruit model. Download provenance stored under assets/ycb/peach.provenance.json.

Bounce table_tennis_ball: real standard-size ball representation, not YCB; diameter40mm, mass2.7g, thin-shell I=2/3 mr², white analytical sphere. Target normal COR0.88, damping ratio0.042/timeconstant0.008 and sliding0.3 are generic-floor engineering priors, NOT ITTF approval measurements. Quadratic drag F=-0.5 rho Cd pi r² |v|v, rho1.225kg/m³,Cd0.47. It is applied only to this object's translational free-joint force at control ticks; does not add energy, excludes wind/lift/spin forces. Existing task launch energy calculation remains ballistic and is approximate under drag. Launch-apex profile0.65–0.85m is a pre-drag target, not a guarantee of actual rebound height.

Default tests use physical_dr_v1 creation-time perturbations, same ranges as current trained tasks, with class fixed and launch reset each episode. This ball's nominal mass is far outside training mass range, so this is a deliberately difficult physical extrapolation test.

```bash
conda activate dcmm
cd /home/isee/Catch/catch_it_copy
bash scripts/test_unseen_objects_v1.sh --episodes 200 --instances 20 --jobs 2
# Roll only: append --only roll_unseen_1119
# Bounce only: append --only bounce_unseen_1119
bash scripts/view_unseen_object_v1.sh roll
bash scripts/view_unseen_object_v1.sh bounce
```

Episodes is PER TARGET. instances are independent fixed physics draws, each runs episodes/instances launches; must divide episodes. CPU inference. The two jobs use synchronous env collections, not20 parallel processes. Results saved in evaluation_results/multitarget_TIMESTAMP. Inspect REPORT.md/summary.csv and COMPLETE. Roll and Bounce success metrics differ (region retention vs0.3s controlled hold).

For nominal-only comparison, append to test command:
`--models-file scripts/eval_unseen_fixed_1119.json`
For nominal viewer append `+benchmark_options.domain_randomization=off` to viewer command (Hydra last override wins). Do not tune physics/rewards using success rates and then call the same data a held-out benchmark.

Server: copy new source/configs and assets/ycb/peach* including its mesh/texture directory. Or download pinned resource:
`python -m ycb_objects.download --objects peach --revision 9e8c6488a2ff673d9aa48a91492fb89423c1b106`
Manifest checkpoint paths can be edited to actual server locations.

Validation: analysis_artifacts/unseen_objects_20261119. Finite rewards, repeated resets preserve sampled mass/inertia/size. Passive20 launches per object for one sampled parameter instance: peach20/20 table-front crossings; pingpong rebound apex0.54–0.66m, apex y0.69–0.94m. Probes disable robot collision and apply the same drag, so these are launch checks, not grasp-success tests or full parameter-range validation. Short policy smoke only; no formal success-rate benchmark performed.

Backup of modified environment/geometry: backups/before_unseen_objects_20261119.
