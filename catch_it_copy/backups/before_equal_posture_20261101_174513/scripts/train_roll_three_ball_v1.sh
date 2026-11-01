#!/usr/bin/env bash
set -euo pipefail
# Activate dcmm first. Default: train Tracking from scratch with legacy reward.
# Stage 2: append task=Catching_TwoStage checkpoint_tracking=/absolute/path/track.pth
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
TRAIN_TASK=Tracking
TRACK_CKPT=''
CATCH_CKPT=''
for TASK_ARG in "$@"; do
  case "$TASK_ARG" in
    task=*) TRAIN_TASK="${TASK_ARG#task=}" ;;
    checkpoint_tracking=*) TRACK_CKPT="${TASK_ARG#checkpoint_tracking=}" ;;
    checkpoint_catching=*) CATCH_CKPT="${TASK_ARG#checkpoint_catching=}" ;;
  esac
done
case "$TRAIN_TASK" in
  Tracking) OUTPUT_NAME=roll_three_ball_tracking_legacy_reward_v1_seed123 ;;
  Catching_TwoStage)
    OUTPUT_NAME=roll_three_ball_catching_legacy_reward_v1_seed123
    if [[ -z "$TRACK_CKPT" && -z "$CATCH_CKPT" ]]; then
      echo 'Stage 2 requires checkpoint_tracking=/absolute/path/to/roll_track.pth (or a compatible Catch checkpoint for continuation).' >&2
      exit 2
    fi ;;
  *) echo 'This script preserves the Tracking -> Catching_TwoStage workflow; task must be Tracking or Catching_TwoStage.' >&2; exit 2 ;;
esac
for TASK_CKPT in "$TRACK_CKPT" "$CATCH_CKPT"; do
  if [[ -n "$TASK_CKPT" && ! -f "$TASK_CKPT" ]]; then
    echo "Checkpoint file does not exist: $TASK_CKPT" >&2
    exit 2
  fi
done
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
python train_DCMM.py \
  test=False task=Tracking num_envs=16 device_id=0 \
  object_motion=roll roll_physics=physical_v1 \
  roll_reward_version=legacy roll_wait_guard=false \
  roll_ball_mode=mixed \
  'roll_ball_weights={original:0.6,tennis_ball:0.2,baseball:0.2}' \
  'roll_speed=[0.8,1.3]' \
  roll_material_file="$TASK_ROOT/ycb_objects/calibration_data/balls.roll_table_priors_v1.json" \
  test_object=default object_eval=false viewer=false \
  checkpoint_tracking='' checkpoint_catching='' \
  output_name="$OUTPUT_NAME" seed=123 \
  "$@"
