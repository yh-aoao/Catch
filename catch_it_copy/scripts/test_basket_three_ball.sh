#!/usr/bin/env bash
set -euo pipefail
# Usage: bash scripts/test_basket_three_ball.sh original|tennis_ball|baseball [Hydra overrides]
# Tennis needs an explicit smaller-ball starting grip:
# bash scripts/test_basket_three_ball.sh tennis_ball 'basket_fixed_hand_flex=[1.0,0.9,0.9]'
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BALL="${1:-original}"
if (($#)); then shift; fi
case "$BALL" in
  original) TEST_OBJECT=default ;;
  tennis_ball|baseball) TEST_OBJECT="ycb:$BALL" ;;
  *) echo 'Ball must be original, tennis_ball or baseball.' >&2; exit 2 ;;
esac
MODEL_RUN="$TASK_ROOT/outputs/BasketAimCarry/2026-10-29/01:23:05"
CHECKPOINT="$MODEL_RUN/nn/throw_basket_catch_best_reward_38.38.pth"
if [[ ! -f "$CHECKPOINT" || ! -f "$MODEL_RUN/resolved_config.yaml" ]]; then
  echo "Missing Basket checkpoint or resolved_config.yaml: $MODEL_RUN" >&2
  exit 1
fi
cd "$TASK_ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python train_DCMM.py \
  --config-path "$MODEL_RUN" --config-name resolved_config \
  hydra.job.chdir=false hydra.run.dir=. \
  test=True num_envs=1 train.ppo.num_actors=1 vector_mode=sync \
  device_id=-1 viewer=true imshow_cam=false seed=0 \
  basket_viewer_speed=1.0 basket_log=true \
  test_object="$TEST_OBJECT" object_eval=false \
  checkpoint_tracking='' checkpoint_catching="$CHECKPOINT" \
  train.ppo.test_num_episodes=30 output_name="basket_video_$BALL" \
  "$@"
