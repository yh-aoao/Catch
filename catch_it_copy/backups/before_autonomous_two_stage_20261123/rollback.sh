#!/usr/bin/env bash
set -euo pipefail
backup_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$backup_dir/../.." && pwd)"
for name in environment.py policy.py pool.py run.py README.md; do
  cp -- "$backup_dir/basket_autonomous/$name" "$project_dir/basket_autonomous/$name"
done
