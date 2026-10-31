#!/usr/bin/env bash
set -euo pipefail
# Usage: bash scripts/test_roll_catch_252.sh original|tennis_ball|baseball [Hydra overrides]
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BALL="${1:-original}"
if (($#)); then shift; fi
case "$BALL" in
  original) TEST_OBJECT=default ;;
  tennis_ball|baseball) TEST_OBJECT="ycb:$BALL" ;;
  *) echo 'Ball must be original, tennis_ball or baseball.' >&2; exit 2 ;;
esac
CHECKPOINT="$TASK_ROOT/../best_model/roll_catch_best_reward_252.54.pth"
if [[ ! -f "$CHECKPOINT" ]]; then echo "Missing checkpoint: $CHECKPOINT" >&2; exit 1; fi
cd "$TASK_ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python train_DCMM.py \
  hydra.job.chdir=false hydra.run.dir=. \
  test=True task=Catching_TwoStage num_envs=1 train.ppo.num_actors=1 vector_mode=sync \
  device_id=-1 viewer=true imshow_cam=false seed=0 \
  object_motion=roll roll_physics=physical_v1 \
  roll_reward_version=legacy roll_wait_guard=false \
  roll_ball_mode=fixed roll_ball_weights=null \
  roll_material_file="$TASK_ROOT/ycb_objects/calibration_data/balls.roll_table_priors_v1.json" \
  'roll_speed=[0.8,1.3]' test_object="$TEST_OBJECT" object_eval=false \
  checkpoint_tracking='' checkpoint_catching="$CHECKPOINT" \
  output_name="roll_catch_252_video_$BALL" \
  "$@"
