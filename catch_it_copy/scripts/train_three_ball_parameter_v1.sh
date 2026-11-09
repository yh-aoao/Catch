#!/usr/bin/env bash
set -euo pipefail
# Matched three-ball experiments. Only physical randomization differs dr/fixed.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if (($# < 2)); then echo 'Usage: train_three_ball_parameter_v1.sh roll|bounce track|catch [dr|fixed] [Hydra overrides]' >&2; exit 2; fi
MOTION="$1"; STAGE="$2"; shift 2
VARIANT=dr
if (($#)) && [[ "$1" == dr || "$1" == fixed ]]; then VARIANT="$1"; shift; fi
case "$MOTION" in roll|bounce) ;; *) echo 'Use roll or bounce' >&2; exit 2 ;; esac
case "$VARIANT" in dr) DR=physical_dr_v1 ;; fixed) DR=off ;; esac
case "$STAGE" in
 track) ENTRY=train_all_targets_track_v1.sh; NAME=tracking ;;
 catch)
  ENTRY=train_all_targets_catch_v1.sh; NAME=catching; TRACK=''
  for ARG in "$@"; do case "$ARG" in checkpoint_tracking=*) TRACK="${ARG#checkpoint_tracking=}" ;; esac; done
  [[ -n "$TRACK" && -f "$TRACK" ]] || { echo 'Catch requires checkpoint_tracking=/path/to/three_ball_track.pth' >&2; exit 2; } ;;
 *) echo 'Use track or catch' >&2; exit 2 ;;
esac
exec bash "$ROOT/scripts/$ENTRY" "$MOTION" \
 num_envs=24 benchmark_catalog="$ROOT/ycb_objects/calibration_data/object_benchmark_three_ball_control_v1.json" \
 "+benchmark_options.domain_randomization=$DR" \
 output_name="${MOTION}_three_ball_${NAME}_parameter_${VARIANT}_v1_seed123" "$@"
