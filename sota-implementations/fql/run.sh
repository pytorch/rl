#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
export MUJOCO_GL=egl HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=1
export WANDB_MODE=disabled WANDB_SILENT=true
exec .venv/bin/python sota-implementations/fql/fql.py "$@"
