#!/bin/bash
#SBATCH --job-name=grpo-sync-multi-node
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

# Set up Ray cluster configuration
export HEAD_NODE=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
export HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -I | awk '{for(i=1;i<=NF;i++) if ($i ~ /^[0-9]+\./) {print $i; exit}}')
export RAY_PORT=6379

# Environment variables
export LIST_TO_STACK=1
export VLLM_USE_V1=0
export RAY_CLUSTER_MANAGED_EXTERNALLY=1

# Run command in Ray cluster
CMD="python grpo-sync.py mode=sync train_model.num_devices=8 ref_model.num_devices=4 inference_model.num_devices=4"
srun bash run_in_ray_cluster.sh "$CMD"

echo "Job completed"
