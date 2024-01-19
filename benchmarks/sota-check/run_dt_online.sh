#!/bin/bash

#SBATCH --job-name=dt_online
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=dt_online_output_%j.txt
#SBATCH --error=dt_online_error_%j.txt

current_commit=$(git rev-parse HEAD)
project_name="torchrl-example-check-$current_commit"
python ../../examples/decision_transformer/dt_online.py \
  optim.pretrain_gradient_steps=55 \
  optim.updates_per_episode=3 \
  optim.warmup_steps=10 \
  optim.device=cuda:0 \
  logger.backend=wandb \
  logger.project_name="$project_name" \
  logger.group_name="dt_online"
