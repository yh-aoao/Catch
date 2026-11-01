#!/usr/bin/env bash
set -euo pipefail
# Stage 2 only: keep the trained Track initialization, learn a new hand policy.
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TRACK_REL="bounce_three_ball_tracking_v1_seed123/2026-10-28/00:42:24/nn/bounce_track_best_reward_19.31.pth"
TRACK_CHECKPOINT="${BOUNCE_TRACK_CHECKPOINT:-}"
CATCH_CHECKPOINT=''
if [[ -z "$TRACK_CHECKPOINT" ]]; then
  for CANDIDATE in "$TASK_ROOT/outputs/$TRACK_REL" "$(dirname -- "$TASK_ROOT")/$TRACK_REL"; do
    if [[ -f "$CANDIDATE" ]]; then TRACK_CHECKPOINT="$CANDIDATE"; break; fi
  done
fi
for ARGUMENT in "$@"; do
  case "$ARGUMENT" in
    checkpoint_tracking=*) TRACK_CHECKPOINT="${ARGUMENT#checkpoint_tracking=}" ;;
    checkpoint_catching=*) CATCH_CHECKPOINT="${ARGUMENT#checkpoint_catching=}" ;;
    task=*)
      if [[ "$ARGUMENT" != task=Catching_TwoStage ]]; then
        echo 'This script trains stage 2 only: task=Catching_TwoStage.' >&2; exit 2
      fi ;;
  esac
done
if [[ -n "$CATCH_CHECKPOINT" ]]; then
  if [[ ! -f "$CATCH_CHECKPOINT" ]]; then
    echo "Catch checkpoint not found: $CATCH_CHECKPOINT" >&2; exit 1
  fi
elif [[ -z "$TRACK_CHECKPOINT" || ! -f "$TRACK_CHECKPOINT" ]]; then
  echo 'Track checkpoint not found. Supply checkpoint_tracking=/absolute/path/to/bounce_track_best_reward_19.31.pth' >&2
  exit 1
fi
cd "$TASK_ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
python train_DCMM.py \
  test=False task=Catching_TwoStage num_envs=16 device_id=0 \
  object_motion=bounce bounce_physics=physical_v2 \
  bounce_grasp_version=controlled_hold_v1 \
  bounce_launch=baseball_intercept_v1 bounce_ball_mode=mixed \
  'bounce_ball_weights={original:0.6,tennis_ball:0.2,baseball:0.2}' \
  bounce_material_file="$TASK_ROOT/ycb_objects/calibration_data/balls.three_ball_training_v1.json" \
  test_object=default object_eval=false viewer=false imshow_cam=false \
  checkpoint_tracking="$TRACK_CHECKPOINT" checkpoint_catching='' \
  'train.ppo.action_catch_denorm=[1.5,0.025,0.08]' \
  bounce_grasp_training.track_lr_scale=0.1 \
  bounce_grasp_training.freeze_track_steps=500000 \
  bounce_grasp_training.freeze_track_normalization=true \
  train.ppo.max_agent_steps=25000000 seed=123 \
  output_name=bounce_three_ball_catching_controlled_hold_v1_seed123 \
  "$@"
