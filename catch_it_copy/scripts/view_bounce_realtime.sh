#!/usr/bin/env bash
set -euo pipefail
# Exact original-physics trajectories, precomputed before 1x viewer playback.
# Usage: bash scripts/view_bounce_realtime.sh [original|tennis_ball|baseball] [options]
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BALL=original
if (($#)) && [[ "$1" != --* ]]; then BALL="$1"; shift; fi
case "$BALL" in
  original|tennis_ball|baseball) ;;
  *) echo 'Ball must be original, tennis_ball or baseball.' >&2; exit 2 ;;
esac
cd "$TASK_ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
python -u -m ycb_objects.bounce_viewer_replay --object "$BALL" "$@"
