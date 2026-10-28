#!/usr/bin/env bash
set -euo pipefail
# Viewer evaluation of the trained three-ball Catch model. Activate dcmm first.
# Usage: bash scripts/test_bounce_three_ball_catch_v1.sh original|tennis_ball|baseball [Hydra overrides]
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_RUN="$(dirname -- "$TASK_ROOT")/bounce_three_ball_catching_v1_seed123/2026-10-28/11:36:32"
BALL="${1:-original}"
if (($#)); then shift; fi
case "$BALL" in
  original) TEST_OBJECT=default ;;
  tennis_ball|baseball) TEST_OBJECT="ycb:$BALL" ;;
  *) echo 'Usage: test_bounce_three_ball_catch_v1.sh original|tennis_ball|baseball [Hydra overrides]' >&2; exit 2 ;;
esac
CHECKPOINT="$MODEL_RUN/nn/bounce_catch_best_reward_288.46.pth"
for REQUIRED_FILE in "$MODEL_RUN/resolved_config.yaml" "$CHECKPOINT"; do
  if [[ ! -f "$REQUIRED_FILE" ]]; then
    echo "Missing model/config: $REQUIRED_FILE" >&2
    exit 1
  fi
done
cd "$TASK_ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
# Saved resolved_config omits Hydra's run directory. Keep relative XML assets
# resolved from TASK_ROOT, as they are with the original configs/config.yaml.
python train_DCMM.py \
  --config-path "$MODEL_RUN" --config-name resolved_config \
  hydra.job.chdir=false hydra.run.dir=. \
  test=True task=Catching_TwoStage num_envs=1 train.ppo.num_actors=1 \
  device_id=-1 viewer=true imshow_cam=false seed=0 \
  object_motion=bounce bounce_physics=physical_v2 \
  bounce_launch=baseball_intercept_v1 \
  bounce_ball_mode=fixed bounce_ball_weights=null \
  bounce_material_file="$TASK_ROOT/ycb_objects/calibration_data/balls.three_ball_training_v1.json" \
  test_object="$TEST_OBJECT" object_eval=false \
  checkpoint_tracking='' checkpoint_catching="$CHECKPOINT" \
  output_name="bounce_catch_video_$BALL" \
  "$@"
