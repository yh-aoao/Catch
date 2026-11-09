#!/usr/bin/env bash
set -euo pipefail
# Full Catch transfer, deliberately not a resume and not Track initialization.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MOTION="${1:-}"; if (($#)); then shift; fi
VARIANT=dr
if (($#)) && [[ "$1" == dr || "$1" == fixed ]]; then VARIANT="$1"; shift; fi
case "$MOTION" in
 roll) MODEL="$ROOT/../11-5/roll_catch_best_reward_270.37.pth" ;;
 bounce) MODEL="$ROOT/../11-5/bounce_catch_best_reward_13.68.pth" ;;
 *) echo 'Usage: finetune_three_ball_catch_parameter_v1.sh roll|bounce [dr|fixed] [Hydra overrides]' >&2; exit 2 ;;
esac
NUM=24; HORIZON=64
for ARG in "$@"; do
 case "$ARG" in
  checkpoint_catching=*) MODEL="${ARG#checkpoint_catching=}" ;;
  checkpoint_tracking=*) [[ "$ARG" == checkpoint_tracking= ]] || { echo 'Use full Catch checkpoint, not Track' >&2; exit 2; } ;;
  task=*) [[ "$ARG" == task=Catching_TwoStage ]] || exit 2 ;;
  num_envs=*) NUM="${ARG#num_envs=}" ;;
  train.ppo.horizon_length=*) HORIZON="${ARG#train.ppo.horizon_length=}" ;;
 esac
done
[[ -n "$MODEL" && -f "$MODEL" ]] || { echo "Missing Catch model: $MODEL" >&2; exit 2; }
[[ "$NUM" =~ ^[1-9][0-9]*$ && "$HORIZON" =~ ^[1-9][0-9]*$ ]] || exit 2
BATCH=$((NUM * HORIZON)); ((BATCH % 2 == 0)) || exit 2
DR=physical_dr_v1; [[ "$VARIANT" != fixed ]] || DR=off
exec bash "$ROOT/scripts/train_object_benchmark.sh" "$MOTION" \
 task=Catching_TwoStage catch_warm_start=true checkpoint_tracking='' checkpoint_catching="$MODEL" hydra.job.chdir=false \
 benchmark_catalog="$ROOT/ycb_objects/calibration_data/object_benchmark_three_ball_control_v1.json" \
 "+benchmark_options.domain_randomization=$DR" \
 num_envs="$NUM" train.ppo.horizon_length="$HORIZON" train.ppo.minibatch_size="$((BATCH / 2))" \
 train.ppo.learning_rate=0.0001 train.ppo.max_agent_steps=1000000000 \
 output_name="${MOTION}_three_ball_catching_parameter_${VARIANT}_from_fixed_v1_seed123" "$@"
