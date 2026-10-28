#!/usr/bin/env bash
set -euo pipefail
# Usage: bash scripts/test_bounce_three_ball_hold_v1.sh /absolute/new_catch.pth original|tennis_ball|baseball [overrides]
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if (($# < 2)); then
  echo 'Usage: test_bounce_three_ball_hold_v1.sh /absolute/new_catch.pth original|tennis_ball|baseball [Hydra overrides]' >&2
  exit 2
fi
CHECKPOINT="$1"
BALL="$2"
shift 2
if [[ ! -f "$CHECKPOINT" ]]; then echo "Missing checkpoint: $CHECKPOINT" >&2; exit 1; fi
CHECKPOINT="$(realpath -- "$CHECKPOINT")"
MODEL_RUN="$(dirname -- "$(dirname -- "$CHECKPOINT")")"
if [[ ! -f "$MODEL_RUN/resolved_config.yaml" ]]; then
  echo "Missing saved config: $MODEL_RUN/resolved_config.yaml" >&2; exit 1
fi
case "$BALL" in
  original) TEST_OBJECT=default ;;
  tennis_ball|baseball) TEST_OBJECT="ycb:$BALL" ;;
  *) echo 'Ball must be original, tennis_ball or baseball.' >&2; exit 2 ;;
esac
cd "$TASK_ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
python train_DCMM.py \
  --config-path "$MODEL_RUN" --config-name resolved_config \
  hydra.job.chdir=false hydra.run.dir=. \
  test=True task=Catching_TwoStage num_envs=1 train.ppo.num_actors=1 \
  device_id=-1 viewer=true imshow_cam=false seed=0 \
  object_motion=bounce bounce_physics=physical_v2 \
  bounce_grasp_version=controlled_hold_v1 \
  bounce_launch=baseball_intercept_v1 bounce_ball_mode=fixed bounce_ball_weights=null \
  bounce_material_file="$TASK_ROOT/ycb_objects/calibration_data/balls.three_ball_training_v1.json" \
  test_object="$TEST_OBJECT" object_eval=false \
  checkpoint_tracking='' checkpoint_catching="$CHECKPOINT" \
  output_name="bounce_hold_video_$BALL" \
  "$@"
