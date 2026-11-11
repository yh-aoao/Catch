#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if (($# < 2)); then echo 'Usage: train_all_targets_episode_dr_v1.sh roll|bounce track|catch|finetune [Hydra overrides]' >&2; exit 2; fi
MOTION="$1"; STAGE="$2"; shift 2
case "$MOTION" in
 roll) NUM=28; PREFIX=roll_seven_object; MODEL="$ROOT/../11-11/roll_catch_best_reward_231.61.pth" ;;
 bounce) NUM=24; PREFIX=bounce_six_ball; MODEL="$ROOT/../11-11/bounce_catch_best_reward_4.84.pth" ;;
 *) echo 'Use roll or bounce' >&2; exit 2 ;;
esac
DR='+benchmark_options.domain_randomization=physical_dr_episode_v1'
case "$STAGE" in
 track)
  exec bash "$ROOT/scripts/train_all_targets_track_v1.sh" "$MOTION" "$DR" \
   hydra.job.chdir=false output_name="${PREFIX}_tracking_episode_dr_v1_seed123" "$@" ;;
 catch)
  TRACK=''
  for ARG in "$@"; do case "$ARG" in checkpoint_tracking=*) TRACK="${ARG#checkpoint_tracking=}" ;; esac; done
  [[ -n "$TRACK" && -f "$TRACK" ]] || { echo 'Catch requires checkpoint_tracking=/path/to/track.pth; use finetune for a full Catch checkpoint.' >&2; exit 2; }
  exec bash "$ROOT/scripts/train_all_targets_catch_v1.sh" "$MOTION" "$DR" \
   hydra.job.chdir=false output_name="${PREFIX}_catching_episode_dr_v1_seed123" "$@" ;;
 finetune)
  HORIZON=64
  for ARG in "$@"; do
   case "$ARG" in
    checkpoint_catching=*) MODEL="${ARG#checkpoint_catching=}" ;;
    checkpoint_tracking=*) [[ "$ARG" == checkpoint_tracking= ]] || { echo 'finetune requires full Catch, not Track' >&2; exit 2; } ;;
    num_envs=*) NUM="${ARG#num_envs=}" ;;
    train.ppo.horizon_length=*) HORIZON="${ARG#train.ppo.horizon_length=}" ;;
    task=*) [[ "$ARG" == task=Catching_TwoStage ]] || exit 2 ;;
   esac
  done
  [[ -f "$MODEL" ]] || { echo "Missing Catch checkpoint: $MODEL" >&2; exit 2; }
  [[ "$NUM" =~ ^[1-9][0-9]*$ && "$HORIZON" =~ ^[1-9][0-9]*$ ]] || exit 2
  BATCH=$((NUM * HORIZON)); ((BATCH % 2 == 0)) || exit 2
  exec bash "$ROOT/scripts/train_object_benchmark.sh" "$MOTION" task=Catching_TwoStage \
   benchmark_catalog="$ROOT/ycb_objects/calibration_data/object_benchmark_all_targets_v1.json" "$DR" \
   catch_warm_start=true checkpoint_tracking='' checkpoint_catching="$MODEL" hydra.job.chdir=false \
   num_envs="$NUM" train.ppo.horizon_length="$HORIZON" train.ppo.minibatch_size="$((BATCH / 2))" \
   train.ppo.learning_rate=0.0001 train.ppo.max_agent_steps=1000000000 \
   output_name="${PREFIX}_catching_episode_dr_from_catch_v1_seed123" "$@" ;;
 *) echo 'Use track, catch, or finetune' >&2; exit 2 ;;
esac
