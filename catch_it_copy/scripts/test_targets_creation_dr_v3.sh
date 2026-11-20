#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if (($#<2)); then echo 'Usage: test_targets_creation_dr_v3.sh roll|bounce /path/to/catch.pth [evaluation options]' >&2; exit 2; fi
MOTION="$1"; CKPT="$2"; shift 2
case "$MOTION" in roll|bounce) ;; *) echo 'Use roll or bounce' >&2; exit 2 ;; esac
# Resolve caller-relative checkpoint before changing directory.
CKPT="$(python -c 'import sys; from pathlib import Path; print(Path(sys.argv[1]).resolve())' "$CKPT")"
cd "$ROOT"
exec python scripts/evaluate_multitarget_catch.py \
 --models-file "$ROOT/scripts/eval_creation_dr_v3.json" --only "${MOTION}_creation_dr_v3" \
 --checkpoint "$CKPT" --catalog "$ROOT/ycb_objects/calibration_data/object_benchmark_creation_v3.json" \
 --episodes 200 --instances 4 --jobs 2 --seed 41000 "$@"
