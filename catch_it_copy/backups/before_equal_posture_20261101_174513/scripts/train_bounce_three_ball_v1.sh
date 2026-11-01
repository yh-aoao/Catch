#!/usr/bin/env bash
set -euo pipefail
# Activate dcmm first. Default is stage 1 Tracking; stage 2 overrides task and loads its checkpoint.
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
python train_DCMM.py \
  test=False task=Tracking num_envs=16 device_id=0 \
  object_motion=bounce bounce_physics=physical_v2 \
  bounce_launch=baseball_intercept_v1 bounce_ball_mode=mixed \
  'bounce_ball_weights={original:0.6,tennis_ball:0.2,baseball:0.2}' \
  bounce_material_file="$TASK_ROOT/ycb_objects/calibration_data/balls.three_ball_training_v1.json" \
  test_object=default object_eval=false viewer=false \
  checkpoint_tracking='' checkpoint_catching='' \
  output_name=bounce_three_ball_tracking_v1_seed123 seed=123 \
  "$@"
