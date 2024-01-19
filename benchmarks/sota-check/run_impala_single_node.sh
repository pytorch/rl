#!/bin/bash

#SBATCH --job-name=impala_1node
#SBATCH --partition=test
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=impala_1node_output_%j.txt
#SBATCH --error=impala_1node_error_%j.txt

python ../../examples/impala/impala_single_node.py \
  collector.total_frames=80 \
  collector.frames_per_batch=20 \
  collector.num_workers=1 \
  logger.test_interval=10 \
  logger.backend=wandb \
  logger.project_name="sota-check"