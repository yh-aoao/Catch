#!/usr/bin/env bash
set -euo pipefail
# First argument: fixed41 or trained mobile-joint checkpoint; remaining args: Hydra overrides.
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="${1:-$TASK_ROOT/../11-1/throw_basket_catch_best_reward_41.30.pth}"
if (($#)); then shift; fi
SOURCE="$(realpath -- "$SOURCE")"
if [[ ! -f "$SOURCE" ]]; then echo "Missing source checkpoint: $SOURCE" >&2; exit 1; fi
cd "$TASK_ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 train_DCMM.py --config-name basket_three_ball \
  test=False num_envs=32 vector_mode=async device_id=0 \
  basket_mixed_source="$SOURCE" checkpoint_catching='' \
  viewer=false basket_log=false basket_self_collision_debug=false \
  train.ppo.max_agent_steps=1000000000000 output_name=BasketThreeBallFixed "$@"
