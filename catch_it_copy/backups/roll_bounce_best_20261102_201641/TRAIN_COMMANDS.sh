#!/usr/bin/env bash
# Recorded recipe. Do not execute both jobs together on one GPU by accident.
# First restore into a fresh directory using verify_restore.py, then:
# conda activate dcmm
# cd /home/isee/Catch/roll_bounce_baseline_restored_20261102

# Roll: from saved Track; change output_name for each new experiment.
bash scripts/train_roll_three_ball_equal_v2.sh \
  task=Catching_TwoStage \
  checkpoint_tracking="$PWD/baseline_models/roll_track_best_reward_30.07.pth" \
  checkpoint_catching='' device_id=0 num_envs=16 seed=123 \
  train.ppo.max_agent_steps=1000000000 \
  output_name=roll_equal_baseline_retrain_20261102

# Bounce: schedule confirmed IN THE CHECKPOINT, overriding script defaults.
bash scripts/train_bounce_three_ball_equal_v2.sh \
  task=Catching_TwoStage \
  checkpoint_tracking="$PWD/baseline_models/bounce_track_best_reward_20.04.pth" \
  checkpoint_catching='' device_id=0 num_envs=16 seed=123 \
  bounce_grasp_training.freeze_track_steps=0 \
  bounce_grasp_training.track_lr_scale=1.0 \
  bounce_grasp_training.freeze_track_normalization=false \
  train.ppo.max_agent_steps=1000000000 \
  output_name=bounce_equal_posture_baseline_retrain_20261102

# 1 billion is a selectable cap, not recovered historical training metadata.
# Exact server configs and stochastic trajectories were not supplied.
