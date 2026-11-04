#!/usr/bin/env bash
set -euo pipefail
# Stage two: initialize Track from a trained checkpoint, initialize Catch afresh.
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MOTION="${1:-}"; if (($#)); then shift; fi
case "$MOTION" in
 roll)
  NUM_ENVS=28; OUTPUT=roll_seven_object_catching_v1_seed123
  TRACK="$TASK_ROOT/outputs/roll_seven_object_tracking_v1_seed123/2026-11-04/01:33:28/nn/roll_track_best_reward_24.83.pth" ;;
 bounce)
  NUM_ENVS=24; OUTPUT=bounce_six_ball_catching_v1_seed123
  TRACK="$TASK_ROOT/outputs/bounce_six_ball_tracking_v1_seed123/2026-11-04/01:33:28/nn/bounce_track_best_reward_19.08.pth" ;;
 *) echo 'Usage: train_all_targets_catch_v1.sh roll|bounce [Hydra overrides]' >&2; exit 2 ;;
esac
HORIZON=64
for ARG in "$@"; do
 case "$ARG" in
  task=*) [[ "$ARG" == task=Catching_TwoStage ]] || { echo 'This entry trains Catching_TwoStage only.' >&2; exit 2; } ;;
  checkpoint_catching=*) [[ "$ARG" == checkpoint_catching= ]] || { echo 'This entry starts Catch from Track; use train_object_benchmark.sh to resume Catch.' >&2; exit 2; } ;;
  checkpoint_tracking=*) TRACK="${ARG#checkpoint_tracking=}" ;;
  num_envs=*) NUM_ENVS="${ARG#num_envs=}" ;;
  train.ppo.horizon_length=*) HORIZON="${ARG#train.ppo.horizon_length=}" ;;
 esac
done
[[ -n "$TRACK" && -f "$TRACK" ]] || { echo "Missing Track checkpoint: $TRACK" >&2; exit 2; }
[[ "$NUM_ENVS" =~ ^[1-9][0-9]*$ && "$HORIZON" =~ ^[1-9][0-9]*$ ]] || { echo 'Use positive integer num_envs/horizon_length' >&2; exit 2; }
BATCH=$((NUM_ENVS * HORIZON))
((BATCH % 2 == 0)) || { echo 'Use an even rollout batch (num_envs * horizon_length).' >&2; exit 2; }
exec bash "$TASK_ROOT/scripts/train_object_benchmark.sh" "$MOTION" \
 task=Catching_TwoStage benchmark_catalog="$TASK_ROOT/ycb_objects/calibration_data/object_benchmark_all_targets_v1.json" \
 num_envs="$NUM_ENVS" train.ppo.horizon_length="$HORIZON" train.ppo.minibatch_size="$((BATCH / 2))" \
 checkpoint_tracking="$TRACK" checkpoint_catching='' train.ppo.max_agent_steps=1000000000 \
 output_name="$OUTPUT" "$@"
