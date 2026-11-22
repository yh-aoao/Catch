#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if (($#<1)); then
  echo 'Usage: bash scripts/test_catch_1122.sh roll|bounce5|bounce6 [evaluation options]' >&2
  exit 2
fi
MODEL="$1"; shift
case "$MODEL" in
  roll) NAME=roll_seven_1122; CATALOG=object_benchmark_creation_v3.json ;;
  bounce5) NAME=bounce_five_1122; CATALOG=object_benchmark_creation_v3.json ;;
  bounce6) NAME=bounce_six_1122; CATALOG=object_benchmark_creation_v2.json ;;
  *) echo 'Choose roll, bounce5 or bounce6' >&2; exit 2 ;;
esac
# Stay in the caller's directory so --checkpoint/--model-dir relative paths
# retain the evaluator's normal meaning. All built-in paths are absolute.
exec python "$ROOT/scripts/evaluate_multitarget_catch.py" \
  --models-file "$ROOT/scripts/eval_catch_1122.json" --only "$NAME" \
  --model-dir "$ROOT/../11-22" \
  --catalog "$ROOT/ycb_objects/calibration_data/$CATALOG" \
  --episodes 200 --instances 4 --jobs 2 --seed 41000 "$@"
