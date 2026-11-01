# Basket 41.30 evaluation, 2026-11-01
Checkpoint: /home/isee/Catch/11-1/throw_basket_catch_best_reward_41.30.pth
Protocol: aim_carry_fingers_v1; arm6_hand12; 35 observations. Saved training steps 4112384, training success meter 0.9820225.

CPU, deterministic policy inference, seed=123, sync single environment, fixed base, physical settling, 100 episodes per completed run. No training or parameter updates. Preset basket_finger_sync_scratch with explicit checkpoint_catching override. Raw logs and summary.json in this directory.

| Object | Grip | Safe scores | Valid launches | Collision episodes |
|---|---|---:|---:|---:|
| Original | Original | 100/100 | 100/100 | 0 |
| Baseball | Original | 0/100 | 100/100 | 0 |
| Tennis | Original | 0/1 completed, then reset preparation collision | 1/1 | 0 completed episodes |
| Tennis | long-finger flex [1.0,0.9,0.9] | 0/100 | 91/100 | 0 |

Tennis adapted grip is a supplementary intervention, not unchanged-policy-and-grip zero-shot transfer. Original tennis protocol could not complete its 100-episode evaluation. Do not report it as 0/100. All completed failed episodes ended ball_on_floor. Reset collision is separate from policy collision episodes.

YCB tests use sphere collision with object-specific visual, radius, mass, inertia and friction priors from ycb_objects/calibration_data/balls.roll_table_priors_v1.json. These are simulation priors, not measured real-ball results. Original radius .04m, mass .05kg; tennis radius .0332758m mass .058kg; baseball radius .0361762m mass .138kg. Single seed and fixed target do not establish robust universal success.

Reproduce from catch_it_copy (replace OBJECT with default/ycb:baseball/ycb:tennis_ball; tennis supplemental adds 'basket_fixed_hand_flex=[1.0,0.9,0.9]'):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 train_DCMM.py --config-name basket_finger_sync_scratch hydra.job.chdir=false test=True num_envs=1 vector_mode=sync device_id=-1 viewer=false basket_log=true checkpoint_catching=/home/isee/Catch/11-1/throw_basket_catch_best_reward_41.30.pth train.ppo.test_num_episodes=100 seed=123 test_object=OBJECT output_name=Basket41Eval
```

Recommended future implementation: stable grip preparation for each ball; fixed-base balanced multi-ball curriculum with gradual physical parameter randomization and original-task rehearsal; then mobile carry/parking/settle/throw using preserved throw skill. Current test_object path is evaluation-only. Existing mobile preset does not yet preserve the fixed physical/carry/finger protocol; cannot directly use this checkpoint as a mobile test checkpoint.
