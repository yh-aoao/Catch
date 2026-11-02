#!/usr/bin/env bash
set -euo pipefail
# No arguments: train from scratch. Optional first checkpoint argument enables explicit warm start.
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE=""
if (($#)) && [[ "$1" != *=* ]]; then SOURCE="$1"; shift; fi
if [[ -n "$SOURCE" ]]; then
  SOURCE="$(realpath -- "$SOURCE")"
  if [[ ! -f "$SOURCE" ]]; then echo "Missing source checkpoint: $SOURCE" >&2; exit 1; fi
fi
cd "$TASK_ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 train_DCMM.py --config-name basket_three_ball \
  test=False num_envs=32 vector_mode=async device_id=0 \
  basket_mixed_source="$SOURCE" checkpoint_catching='' \
  viewer=false basket_log=false basket_self_collision_debug=false \
  train.ppo.max_agent_steps=1000000000000 output_name=BasketThreeBallScratch "$@"
