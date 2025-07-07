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
export HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -i | head -n 1)
export RAY_PORT=6379

# Environment variables for the application
export VLLM_USE_V1=0

# Indicate that Ray cluster is managed externally
export RAY_CLUSTER_MANAGED_EXTERNALLY=1

# Optional: Set Ray-specific environment variables
export RAY_DEDUP_LOGS=0  # Avoid duplicate logs
export PYTHONUNBUFFERED=1  # Ensure Python output is not buffered

echo "Starting Ray cluster with head node: $HEAD_NODE ($HEAD_NODE_IP)"
echo "Total nodes: $SLURM_NNODES"
echo "Job ID: $SLURM_JOB_ID"

CMD="python sota-implementations/grpo/grpo-sync.py mode=sync train_model.num_devices=8 ref_model.num_devices=4 inference_model.num_devices=4"

# Run the Ray cluster
srun bash run_in_ray_cluster.sh "$CMD"

echo "Job completed"
