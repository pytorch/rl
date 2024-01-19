#!/bin/bash

#SBATCH --job-name=iql_online
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=iql_online_output_%j.txt
#SBATCH --error=iql_online_error_%j.txt

current_commit=$(git rev-parse HEAD)
project_name="sota-check_$current_commit"
python ../../examples/iql/iql_online.py \
  collector.total_frames=256 \
  optim.batch_size=10 \
  collector.frames_per_batch=16 \
  env.train_num_envs=1 \
  optim.device=cuda:0 \
  collector.device=cuda:0 \
  logger.mode=offline \
  logger.backend=wandb \
  logger.project_name="$project_name"
