#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MOTION="${1:-}"; (($#==0)) || shift
case "$MOTION" in
 roll) OBJ=peach; CKPT="$ROOT/../11-19/roll_catch_best_reward_376.90.pth"; EXTRA=(roll_palm_version=palm_z_v1 roll_ball_mode=fixed) ;;
 bounce) OBJ=table_tennis_ball; CKPT="$ROOT/../11-19/bounce_catch_best_reward_2.11.pth"; EXTRA=(bounce_ball_mode=fixed) ;;
 *) echo 'Usage: view_unseen_object_v1.sh roll|bounce [checkpoint.pth] [Hydra overrides]' >&2; exit 2 ;;
esac
if (($#)) && [[ "$1" != *=* ]]; then CKPT="$1"; shift; fi
[[ -f "$CKPT" ]] || { echo "Missing checkpoint: $CKPT" >&2; exit 2; }
exec bash "$ROOT/scripts/train_object_benchmark.sh" "$MOTION" \
 test=True task=Catching_TwoStage num_envs=1 train.ppo.num_actors=1 vector_mode=sync \
 device_id=-1 viewer=true imshow_cam=false hydra.job.chdir=false \
 benchmark_catalog="$ROOT/ycb_objects/calibration_data/object_benchmark_unseen_v1.json" \
 '+benchmark_options.domain_randomization=physical_dr_v1' "${EXTRA[@]}" \
 test_object="$OBJ" checkpoint_tracking='' checkpoint_catching="$CKPT" \
 seed="$RANDOM" output_name="${MOTION}_unseen_viewer" "$@"
