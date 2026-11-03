#!/usr/bin/env bash
set -euo pipefail
# Usage: view_object_benchmark.sh roll|bounce object [checkpoint.pth] [options]
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if (($# < 2)); then echo 'Usage: view_object_benchmark.sh roll|bounce object [checkpoint.pth] [options]' >&2; exit 2; fi
MOTION="$1"; OBJECT="$2"; shift 2
case "$MOTION:$OBJECT" in
  roll:orange|roll:apple|roll:lemon|roll:tomato_soup_can)
    CHECKPOINT="$(dirname -- "$TASK_ROOT")/11-3-2/roll_catch_best_reward_269.97.pth" ;;
  bounce:racquetball|bounce:golf_ball|bounce:softball)
    CHECKPOINT="$(dirname -- "$TASK_ROOT")/11-3-2/bounce_catch_best_reward_9.40.pth" ;;
  *) echo 'Choose roll orange/apple/lemon/tomato_soup_can or bounce racquetball/golf_ball/softball' >&2; exit 2 ;;
esac
if (($#)) && [[ "$1" != --* && "$1" != *=* ]]; then CHECKPOINT="$1"; shift; fi
[[ -f "$CHECKPOINT" ]] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 1; }
cd "$TASK_ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
if [[ "$MOTION" == bounce ]]; then
  exec python -u -m ycb_objects.bounce_viewer_replay --object "$OBJECT" \
    --checkpoint "$CHECKPOINT" --config "$TASK_ROOT/configs/benchmark_bounce_viewer_v1.yaml" \
    --episodes 5 --seed 0 --speed 1 "$@"
else
  exec python train_DCMM.py test=True task=Catching_TwoStage \
    num_envs=1 train.ppo.num_actors=1 vector_mode=sync device_id=-1 \
    viewer=true imshow_cam=false object_motion=roll object_eval=false \
    object_benchmark=roll_v1 roll_physics=physical_v1 \
    roll_reward_version=legacy roll_wait_guard=false \
    roll_ball_mode=fixed roll_ball_weights=null roll_material_file=null \
    roll_speed=null test_object="ycb:$OBJECT" \
    checkpoint_tracking='' checkpoint_catching="$CHECKPOINT" \
    output_name="roll_benchmark_video_$OBJECT" seed=0 "$@"
fi
