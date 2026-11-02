#!/usr/bin/env bash
set -euo pipefail
# New reward, three balls equally sampled, conditional on true radius/mass, from scratch.
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 train_DCMM.py --config-name basket_three_ball_ballistic \
 test=False num_envs=32 vector_mode=async device_id=0 \
 basket_mixed_source='' checkpoint_catching='' checkpoint_tracking='' basket_warm_start='' \
 viewer=false basket_log=false basket_self_collision_debug=false \
 train.ppo.max_agent_steps=1000000000000 "$@"
