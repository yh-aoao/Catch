#!/usr/bin/env bash
set -euo pipefail
# Usage: bash scripts/train_basket_single.sh original|tennis_ball|baseball [Hydra overrides]
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BALL="${1:-original}"
if (($#)); then shift; fi
case "$BALL" in
 original) CONFIG=basket_single_original ;;
 tennis_ball|tennis) CONFIG=basket_single_tennis ;;
 baseball) CONFIG=basket_single_baseball ;;
 *) echo 'Choose original, tennis_ball, or baseball.' >&2;exit 2 ;;
esac
cd "$TASK_ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 train_DCMM.py --config-name "$CONFIG" \
 test=False num_envs=32 vector_mode=async device_id=0 \
 basket_mixed_source='' checkpoint_catching='' checkpoint_tracking='' basket_warm_start='' \
 viewer=false basket_log=false basket_self_collision_debug=false \
 train.ppo.max_agent_steps=1000000000000 "$@"
