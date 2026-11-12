#!/usr/bin/env bash
set -euo pipefail
BACKUP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_ROOT="$(cd -- "$BACKUP_DIR/../.." && pwd)"
cp -- "$BACKUP_DIR/basket_fingers.py.original" "$TASK_ROOT/gym_dcmm/utils/basket_fingers.py"
