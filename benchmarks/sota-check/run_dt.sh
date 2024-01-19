#!/bin/bash

#SBATCH --job-name=dt
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=dt_offline_output_%j.txt
#SBATCH --error=dt_offline_error_%j.txt

current_commit=$(git rev-parse HEAD)
project_name="sota-check_$current_commit"
python ../../examples/dt/dt.py \
  optim.pretrain_gradient_steps=55 \
  optim.updates_per_episode=3 \
  optim.warmup_steps=10 \
  optim.device=cuda:0 \
  env.backend=gymnasium \
  env.name=HalfCheetah-v4 \
  logger.backend=wandb \
  logger.project_name="$project_name"
