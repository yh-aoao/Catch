#!/usr/bin/env bash
set -euo pipefail
# Stage 1 by default. Stage 2 requires task=Catching_TwoStage and a checkpoint.
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_TASK=Tracking
for ARGUMENT in "$@"; do
  case "$ARGUMENT" in task=*) TRAIN_TASK="${ARGUMENT#task=}" ;; esac
done
case "$TRAIN_TASK" in
  Tracking) OUTPUT_NAME=roll_three_ball_tracking_equal_v2_seed123 ;;
  Catching_TwoStage) OUTPUT_NAME=roll_three_ball_catching_equal_v2_seed123 ;;
  *) echo 'task must be Tracking or Catching_TwoStage' >&2; exit 2 ;;
esac
exec bash "$TASK_ROOT/scripts/train_roll_three_ball_v1.sh" \
  task="$TRAIN_TASK" \
  'roll_ball_weights={original:1,tennis_ball:1,baseball:1}' \
  output_name="$OUTPUT_NAME" "$@"
