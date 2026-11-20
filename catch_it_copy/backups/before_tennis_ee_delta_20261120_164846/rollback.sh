#!/usr/bin/env bash
set -euo pipefail
backup_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$backup_dir/../.." && pwd)"
tar -xzf "$backup_dir/code_snapshot.tar.gz" -C "$project_dir" \
  train_DCMM.py \
  gym_dcmm/envs/DcmmVecEnv.py \
  gym_dcmm/envs/basket_mixed.py \
  gym_dcmm/algs/ppo_dcmm/ppo_dcmm_basket_mixed.py \
  gym_dcmm/algs/ppo_dcmm/ppo_dcmm_basket_mobile_joint.py \
  gym_dcmm/algs/ppo_dcmm/ppo_dcmm_catch_one_stage.py \
  gym_dcmm/algs/ppo_dcmm/basket_mobile_joint.py \
  gym_dcmm/algs/ppo_dcmm/basket_mixed_policy.py \
  gym_dcmm/utils/basket_throw_policy.py
