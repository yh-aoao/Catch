#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MOTION="${1:-}"; STAGE="${2:-}"; if (($# >= 2)); then shift 2; else exit 2; fi
case "$MOTION" in roll) PREFIX=roll_seven_object ;; bounce) PREFIX=bounce_six_ball ;; *) echo 'Use roll or bounce' >&2; exit 2 ;; esac
case "$STAGE" in
 track) ENTRY=train_all_targets_track_v1.sh; TASK=tracking ;;
 catch)
  ENTRY=train_all_targets_catch_v1.sh; TASK=catching; TRACK=''
  for ARG in "$@"; do case "$ARG" in checkpoint_tracking=*) TRACK="${ARG#checkpoint_tracking=}" ;; esac; done
  [[ -n "$TRACK" && -f "$TRACK" ]] || { echo 'Catch requires checkpoint_tracking=/path/to/DR_track.pth' >&2; exit 2; } ;;
 *) echo 'Usage: train_all_targets_dr_v1.sh roll|bounce track|catch [Hydra overrides]' >&2; exit 2 ;;
esac
exec bash "$ROOT/scripts/$ENTRY" "$MOTION" \
 '+benchmark_options.domain_randomization=physical_dr_v1' \
 output_name="${PREFIX}_${TASK}_dr_v1_seed123" "$@"
