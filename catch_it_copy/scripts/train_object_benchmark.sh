#!/usr/bin/env bash
set -euo pipefail
# New-object benchmark only; defaults to Tracking from scratch.
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MOTION="${1:-}"; if (($#)); then shift; fi
TRAIN_TASK=Tracking; TRACK=''; CATCH=''
for ARG in "$@"; do
 case "$ARG" in task=*) TRAIN_TASK="${ARG#task=}" ;; checkpoint_tracking=*) TRACK="${ARG#checkpoint_tracking=}" ;; checkpoint_catching=*) CATCH="${ARG#checkpoint_catching=}" ;; esac
done
case "$TRAIN_TASK" in
 Tracking) ;;
 Catching_TwoStage) [[ -n "$TRACK" || -n "$CATCH" ]] || { echo 'Catch requires checkpoint_tracking or checkpoint_catching' >&2; exit 2; } ;;
 *) echo 'Use Tracking or Catching_TwoStage' >&2; exit 2 ;;
esac
for MODEL in "$TRACK" "$CATCH"; do [[ -z "$MODEL" || -f "$MODEL" ]] || { echo "Missing checkpoint: $MODEL" >&2; exit 2; }; done
cd "$TASK_ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
EXTRA=()
case "$MOTION" in
 roll) EXTRA+=(roll_physics=physical_v1 roll_reward_version=legacy roll_wait_guard=false roll_ball_mode=mixed roll_ball_weights=null roll_material_file=null roll_speed=null) ;;
 bounce)
  EXTRA+=(bounce_physics=physical_v2 bounce_ball_mode=mixed bounce_ball_weights=null bounce_material_file=null)
  if [[ "$TRAIN_TASK" == Catching_TwoStage ]]; then
    EXTRA+=(bounce_grasp_version=controlled_hold_v1 '+bounce_grasp.finger_backfold_weight=0.15' '+bounce_grasp.finger_prepare_sync_weight=0.05' '+bounce_grasp.finger_contact_weight=1.0' '+bounce_grasp.thumb_opposition_weight=0.5' 'train.ppo.action_catch_denorm=[1.5,0.025,0.08]' bounce_grasp_training.freeze_track_steps=0 bounce_grasp_training.track_lr_scale=1.0 bounce_grasp_training.freeze_track_normalization=false)
  fi ;;
 *) echo 'Use roll or bounce as first argument' >&2; exit 2 ;;
esac
exec python train_DCMM.py test=False task="$TRAIN_TASK" object_motion="$MOTION" \
  object_benchmark="${MOTION}_v1" test_object=default object_eval=false \
  num_envs=16 device_id=0 viewer=false imshow_cam=false \
  checkpoint_tracking="$TRACK" checkpoint_catching="$CATCH" \
  output_name="${MOTION}_new_objects_${TRAIN_TASK}_v1_seed123" seed=123 \
  "${EXTRA[@]}" "$@"
