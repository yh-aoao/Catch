#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${1:-track}"; (($# == 0)) || shift
case "$STAGE" in
 track) TASK=Tracking ;;
 catch)
 TASK=Catching_TwoStage; TRACK=''
 for ARG in "$@"; do case "$ARG" in checkpoint_tracking=*) TRACK="${ARG#checkpoint_tracking=}" ;; esac; done
 [[ -f "$TRACK" ]] || { echo 'Supply checkpoint_tracking=/absolute/path/to/track.pth' >&2; exit 2; } ;;
 *) echo 'Use track or catch' >&2; exit 2 ;;
esac
cd "$ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
python train_DCMM.py task="$TASK" object_motion=throw throw_generalization=multishape_v1 \
 object_eval=false test_object=default test=false viewer=false imshow_cam=false \
 num_envs=32 seed=123 device_id=0 checkpoint_tracking='' checkpoint_catching='' \
 train.ppo.max_agent_steps=1000000000 hydra.job.chdir=false \
 output_name="throw_multishape_${STAGE}_v1_seed123" "$@"
