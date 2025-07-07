#!/bin/bash
#SBATCH --job-name=grpo-async-multi-node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --exclusive
#SBATCH --output=logs/%x.job%j.out
#SBATCH --time=1:00:00

# Exit on any error
set -euo pipefail

# Ensure logs directory exists
mkdir -p logs

# Environment variables
export LIST_TO_STACK=1
export VLLM_USE_V1=0
export RAY_CLUSTER_MANAGED_EXTERNALLY=1

# Run command in Ray cluster
CMD="python grpo-async.py mode=async train_model.num_devices=8 ref_model.num_devices=4 inference_model.num_devices=4"
srun bash run_in_ray_cluster.sh "$CMD"

echo "Job completed"
