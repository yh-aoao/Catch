#!/usr/bin/env bash
set -euo pipefail
# Seven Roll objects or six Bounce balls, balanced fixed scene slots.
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MOTION="${1:-}"; if (($#)); then shift; fi
case "$MOTION" in
 roll) NUM_ENVS=28; OUTPUT=roll_seven_object_tracking_v1_seed123 ;;
 bounce) NUM_ENVS=24; OUTPUT=bounce_six_ball_tracking_v1_seed123 ;;
 *) echo 'Usage: train_all_targets_track_v1.sh roll|bounce [Hydra overrides]' >&2; exit 2 ;;
esac
HORIZON=64
for ARG in "$@"; do
 case "$ARG" in
  task=*) [[ "$ARG" == task=Tracking ]] || { echo 'This entry trains Tracking only.' >&2; exit 2; } ;;
  num_envs=*) NUM_ENVS="${ARG#num_envs=}" ;;
  train.ppo.horizon_length=*) HORIZON="${ARG#train.ppo.horizon_length=}" ;;
 esac
done
[[ "$NUM_ENVS" =~ ^[1-9][0-9]*$ && "$HORIZON" =~ ^[1-9][0-9]*$ ]] || { echo 'Use positive integer num_envs/horizon_length' >&2; exit 2; }
BATCH=$((NUM_ENVS * HORIZON))
((BATCH % 2 == 0)) || { echo 'Use an even rollout batch (num_envs * horizon_length).' >&2; exit 2; }
MINIBATCH=$((BATCH / 2))
exec bash "$TASK_ROOT/scripts/train_object_benchmark.sh" "$MOTION" \
 task=Tracking benchmark_catalog="$TASK_ROOT/ycb_objects/calibration_data/object_benchmark_all_targets_v1.json" \
 num_envs="$NUM_ENVS" train.ppo.horizon_length="$HORIZON" train.ppo.minibatch_size="$MINIBATCH" \
 checkpoint_tracking='' checkpoint_catching='' train.ppo.max_agent_steps=1000000000 \
 output_name="$OUTPUT" "$@"
