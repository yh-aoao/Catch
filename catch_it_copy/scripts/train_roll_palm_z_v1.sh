#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${1:-catch}"; (($#==0)) || shift
case "$STAGE" in
 track|catch) ;;
 *) echo 'Use track or catch' >&2; exit 2 ;;
esac
exec bash "$ROOT/scripts/train_all_targets_episode_dr_v1.sh" roll "$STAGE" \
 roll_palm_version=palm_z_v1 num_envs=28 \
 output_name="roll_seven_object_${STAGE}_palm_z_v1_seed123" "$@"
