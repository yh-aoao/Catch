#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec python scripts/evaluate_multitarget_catch.py \
 --models-file "$ROOT/scripts/eval_unseen_1119.json" \
 --catalog "$ROOT/ycb_objects/calibration_data/object_benchmark_unseen_v1.json" \
 --episodes 200 --instances 4 --jobs 2 --seed 41000 "$@"
