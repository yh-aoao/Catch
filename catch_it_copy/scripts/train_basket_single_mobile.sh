#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BALL="${1:?Usage: bash scripts/train_basket_single_mobile.sh tennis_ball|baseball [Hydra overrides]}"
shift
case "$BALL" in
  tennis|tennis_ball) CONFIG=basket_mobile_tennis ;;
  baseball) CONFIG=basket_mobile_baseball ;;
  *) echo "Expected tennis_ball or baseball" >&2; exit 1 ;;
esac
cd "$TASK_ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 train_DCMM.py --config-name "$CONFIG" \
  test=False num_envs=32 vector_mode=async device_id=0 \
  checkpoint_catching='' basket_warm_start='' \
  viewer=false basket_log=false basket_self_collision_debug=false \
  train.ppo.max_agent_steps=1000000000000 "$@"
