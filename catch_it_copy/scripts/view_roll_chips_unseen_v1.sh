#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="$ROOT/../11-19/roll_catch_best_reward_376.90.pth"
if (($#)) && [[ "$1" != *=* ]]; then CKPT="$1"; shift; fi
exec bash "$ROOT/scripts/train_object_benchmark.sh" roll \
 test=True task=Catching_TwoStage num_envs=1 train.ppo.num_actors=1 vector_mode=sync \
 device_id=-1 viewer=true imshow_cam=false hydra.job.chdir=false \
 benchmark_catalog="$ROOT/ycb_objects/calibration_data/object_benchmark_chips_unseen_v1.json" \
 '+benchmark_options.domain_randomization=physical_dr_v1' roll_palm_version=palm_z_v1 \
 roll_ball_mode=fixed test_object=chips_can checkpoint_tracking='' checkpoint_catching="$CKPT" \
 seed="$RANDOM" output_name=roll_chips_unseen_viewer "$@"
