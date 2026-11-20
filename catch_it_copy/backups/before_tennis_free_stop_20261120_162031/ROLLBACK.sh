#!/usr/bin/env bash
set -euo pipefail
BACKUP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_ROOT="$(cd -- "$BACKUP_DIR/../.." && pwd)"
cp -- "$BACKUP_DIR/train_DCMM.py" "$TASK_ROOT/train_DCMM.py"
cp -- "$BACKUP_DIR/gym_dcmm/envs/basket_free_stop.py" "$TASK_ROOT/gym_dcmm/envs/basket_free_stop.py"
