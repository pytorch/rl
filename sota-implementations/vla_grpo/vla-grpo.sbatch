#!/bin/bash
# Single-node, single-GPU launcher for the LIBERO configuration of the VLA
# GRPO recipe. The MuJoCo workers are CPU processes; the policy trains on one
# GPU (LoRA fallback) -- size cpus-per-task to
# collector.num_collectors * collector.envs_per_collector + headroom.
# The launch command below overrides the multi-GPU H100 defaults of
# vla_grpo_libero.yaml (train GPU 0, rollout server GPU 1, render GPUs 2-7)
# down to one GPU, as documented in the README hardware notes. With
# env.parallel_group_repeats=true the collector env total must stay at
# len(env.task_ids) * collector.group_size = 80 to cover every task, hence
# one collector with 80 envs. For multi-GPU EGL rendering, request more GPUs
# and override env.render_gpu_ids accordingly.
#SBATCH --job-name=vla-grpo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:1
#SBATCH --output=logs/vla-grpo-%j.out
#SBATCH --error=logs/vla-grpo-%j.err

set -euo pipefail
mkdir -p logs

# headless rendering for the LIBERO (MuJoCo) workers
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
# the vendored OpenVLA-OFT constants (chunk size 8, action dim 7)
export ROBOT_PLATFORM=LIBERO

python sota-implementations/vla_grpo/vla-grpo.py \
  --config-name vla_grpo_libero \
  policy.device=null \
  collector.policy_device=null \
  collector.num_collectors=1 \
  collector.envs_per_collector=80 \
  'env.render_gpu_ids=[0]' \
  'env.eval_render_gpu_ids=[0]' \
  logger.group_name="${SLURM_JOB_ID}" \
  "$@"
