#!/bin/bash

#SBATCH --job-name=redq
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=redq_output_%j.txt
#SBATCH --error=redq_error_%j.txt

current_commit=$(git rev-parse HEAD)
project_name="torchrl-example-check-$current_commit"
python ../../examples/redq/redq.py \
  num_workers=2 \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  collector.frames_per_batch=16 \
  collector.env_per_collector=1 \
  buffer.batch_size=10 \
  buffer.size=120 \
  collector.device=cuda:0 \
  optim.steps_per_batch=1 \
  logger.record_frames=4 \
  logger.backend=wandb \
  logger.project_name="$project_name" \
  logger.group_name="redq"
