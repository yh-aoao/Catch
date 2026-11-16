#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# Usage: bash scripts/test_original_radius.sh roll|bounce|both [checkpoint] [extra evaluator options]
MOTION="${1:-both}"; (($# == 0)) || shift
ARGS=()
case "$MOTION" in
 roll) ARGS+=(--only roll_236) ;;
 bounce) ARGS+=(--only bounce_157) ;;
 both) ;;
 *) echo 'Use roll, bounce, or both' >&2; exit 2 ;;
esac
MANIFEST="$ROOT/scripts/eval_original_radius_1116.json"
TEMP=''
trap '[[ -z "$TEMP" ]] || rm -f -- "$TEMP"' EXIT
if (($#)) && [[ "$1" != --* ]]; then
 [[ "$MOTION" != both ]] || { echo 'Custom checkpoint requires roll or bounce' >&2; exit 2; }
 TEMP="$(mktemp /tmp/catch-radius-model-XXXXXX.json)"
 python - "$MANIFEST" "$TEMP" "$MOTION" "$1" <<'PY'
import json,sys
from pathlib import Path
rows=json.loads(Path(sys.argv[1]).read_text())
rows=[r for r in rows if r['motion']==sys.argv[3]]
rows[0]['checkpoint']=str(Path(sys.argv[4]).resolve())
Path(sys.argv[2]).write_text(json.dumps(rows))
PY
 MANIFEST="$TEMP"; shift
fi
cd "$ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
for SCALE in 1.00 0.95 0.90; do
 python scripts/evaluate_multitarget_catch.py \
  --models-file "$MANIFEST" "${ARGS[@]}" --objects original \
  --radius-scale "$SCALE" --mass-policy fixed \
  --episodes 100 --instances 4 --jobs 2 --seed 41000 \
  --output "$ROOT/evaluation_results/original_radius_${MOTION}_${STAMP}/r${SCALE}" "$@"
done
