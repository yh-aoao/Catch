#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if (($#<2)); then echo 'Usage: train_targets_creation_dr_v2.sh roll|bounce track|catch [overrides]' >&2; exit 2; fi
MOTION="$1"; STAGE="$2"; shift 2
case "$MOTION" in
 roll) PREFIX=roll_seven_object_tuna; EXTRA=(roll_palm_version=palm_z_v1) ;;
 bounce) PREFIX=bounce_six_ball; EXTRA=() ;;
 *) exit 2 ;;
esac
case "$STAGE" in
 track) ENTRY=train_all_targets_track_v1.sh ;;
 catch) ENTRY=train_all_targets_catch_v1.sh; TRACK=''
 for ARG in "$@"; do case "$ARG" in checkpoint_tracking=*) TRACK="${ARG#checkpoint_tracking=}" ;; esac; done
 [[ -f "$TRACK" ]] || { echo 'Supply checkpoint_tracking=/absolute/path/track.pth' >&2; exit 2; } ;;
 *) exit 2 ;;
esac
exec bash "$ROOT/scripts/$ENTRY" "$MOTION" \
 benchmark_catalog="$ROOT/ycb_objects/calibration_data/object_benchmark_creation_v2.json" \
 '+benchmark_options.domain_randomization=physical_dr_v1' "${EXTRA[@]}" \
 output_name="${PREFIX}_${STAGE}_creation_dr_v2_seed123" "$@"
