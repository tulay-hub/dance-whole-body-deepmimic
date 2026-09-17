#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPOSITORY_ROOT="$(cd -- "$PROJECT_ROOT/../.." && pwd)"
FRAMEWORK_ROOT="$REPOSITORY_ROOT/frameworks/shared/lens110_isaaclab/lens110/legged_lab_lbot"

cd "$FRAMEWORK_ROOT"
exec "${PYTHON:-python}" scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac--Deepmimic-Lens110-v0 "$@"
