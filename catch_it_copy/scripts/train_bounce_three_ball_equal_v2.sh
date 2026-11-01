#!/usr/bin/env bash
set -euo pipefail
# Equal-probability Tracking -> controlled Catch with opt-in finger posture costs.
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_TASK=Tracking
TRACK_CHECKPOINT=''
CATCH_CHECKPOINT=''
for ARGUMENT in "$@"; do
  case "$ARGUMENT" in
    task=*) TRAIN_TASK="${ARGUMENT#task=}" ;;
    checkpoint_tracking=*) TRACK_CHECKPOINT="${ARGUMENT#checkpoint_tracking=}" ;;
    checkpoint_catching=*) CATCH_CHECKPOINT="${ARGUMENT#checkpoint_catching=}" ;;
  esac
done
case "$TRAIN_TASK" in
  Tracking)
    exec bash "$TASK_ROOT/scripts/train_bounce_three_ball_v1.sh" \
      task=Tracking 'bounce_ball_weights={original:1,tennis_ball:1,baseball:1}' \
      output_name=bounce_three_ball_tracking_equal_v2_seed123 "$@"
    ;;
  Catching_TwoStage)
    if [[ -z "$TRACK_CHECKPOINT" && -z "$CATCH_CHECKPOINT" ]]; then
      echo 'Supply checkpoint_tracking from the new equal-probability Track run (or checkpoint_catching to resume).' >&2
      exit 2
    fi
    exec bash "$TASK_ROOT/scripts/train_bounce_three_ball_hold_v1.sh" \
      'bounce_ball_weights={original:1,tennis_ball:1,baseball:1}' \
      '+bounce_grasp.finger_backfold_weight=0.15' \
      '+bounce_grasp.finger_prepare_sync_weight=0.05' \
      '+bounce_grasp.finger_contact_weight=1.0' \
      '+bounce_grasp.thumb_opposition_weight=0.5' \
      output_name=bounce_three_ball_catching_equal_posture_v2_seed123 "$@"
    ;;
  *) echo 'task must be Tracking or Catching_TwoStage' >&2; exit 2 ;;
esac
