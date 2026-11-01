#!/usr/bin/env bash
set -euo pipefail
# First argument: fixed-base 41.30 checkpoint. Remaining arguments: Hydra overrides.
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="${1:-$TASK_ROOT/../11-1/throw_basket_catch_best_reward_41.30.pth}"
if (($#)); then shift; fi
CHECKPOINT="$(realpath -- "$CHECKPOINT")"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing fixed-base throwing checkpoint: $CHECKPOINT" >&2
  exit 1
fi
cd "$TASK_ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 train_DCMM.py --config-name basket_mobile41 \
  test=False num_envs=32 vector_mode=async device_id=0 \
  basket_throw_checkpoint="$CHECKPOINT" \
  checkpoint_catching='' basket_warm_start='' \
  viewer=false basket_log=false basket_self_collision_debug=false \
  train.ppo.max_agent_steps=1000000000000 \
  output_name=BasketMobile41 "$@"
