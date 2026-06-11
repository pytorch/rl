#!/bin/bash
# Single-node launcher for the LIBERO configuration of the VLA GRPO recipe.
# The MuJoCo workers are CPU processes; the policy trains on one GPU (LoRA
# fallback) -- size cpus-per-task to env.num_envs + headroom.
#SBATCH --job-name=vla-grpo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --output=logs/vla-grpo-%j.out
#SBATCH --error=logs/vla-grpo-%j.err

set -euo pipefail
mkdir -p logs

# headless rendering for the LIBERO (MuJoCo) workers
export MUJOCO_GL=egl
# the vendored OpenVLA-OFT constants (chunk size 8, action dim 7)
export ROBOT_PLATFORM=LIBERO

python sota-implementations/vla_grpo/vla-grpo.py \
  --config-name vla_grpo_libero \
  logger.group_name="${SLURM_JOB_ID}" \
  "$@"
