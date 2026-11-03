#!/usr/bin/env bash
set -euo pipefail
# Usage: bash scripts/view_roll_bounce_20261103.sh roll|bounce original|tennis_ball|baseball best|last [viewer options]
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if (($# < 3)); then
  echo 'Usage: view_roll_bounce_20261103.sh roll|bounce original|tennis_ball|baseball best|last [options]' >&2
  exit 2
fi
MOTION="$1"; BALL="$2"; VARIANT="$3"; shift 3
case "$BALL" in original|tennis_ball|baseball) ;; *) echo 'Unknown ball' >&2; exit 2 ;; esac
case "$MOTION:$VARIANT" in
  roll:best) MODEL_NAME=roll_catch_best_reward_262.02.pth ;;
  roll:last) MODEL_NAME=roll_catch_last.pth ;;
  bounce:best) MODEL_NAME=bounce_catch_best_reward_8.55.pth ;;
  bounce:last) MODEL_NAME=bounce_catch_last.pth ;;
  *) echo 'Choose roll|bounce and best|last' >&2; exit 2 ;;
esac
CHECKPOINT="$(dirname -- "$TASK_ROOT")/11-3/$MODEL_NAME"
[[ -f "$CHECKPOINT" ]] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 1; }
if [[ "$MOTION" == roll ]]; then
  exec bash "$TASK_ROOT/scripts/test_roll_catch_252.sh" "$BALL" checkpoint_catching="$CHECKPOINT" \
    output_name="roll_catch_20261103_${VARIANT}_${BALL}" "$@"
else
  CONFIG="$(dirname -- "$TASK_ROOT")/analysis_artifacts/catch_best_last_20261103/bounce_config_reconstructed.yaml"
  exec bash "$TASK_ROOT/scripts/view_bounce_realtime.sh" "$BALL" \
    --checkpoint "$CHECKPOINT" --config "$CONFIG" --episodes 10 --seed 0 --speed 1 "$@"
fi
