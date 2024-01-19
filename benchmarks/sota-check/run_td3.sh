#!/bin/bash

#SBATCH --job-name=td3
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=td3_output_%j.txt
#SBATCH --error=td3_error_%j.txt

current_commit=$(git rev-parse HEAD)
project_name="torchrl-example-check-$current_commit"
python ../../examples/td3/td3.py \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  optim.batch_size=10 \
  collector.frames_per_batch=16 \
  collector.num_workers=4 \
  collector.env_per_collector=2 \
  collector.device=cuda:0 \
  collector.device=cuda:0 \
  network.device=cuda:0 \
  env.name=Pendulum-v1 \
  logger.backend=wandb \
  logger.project_name="$project_name" \
  logger.group_name="td3"
