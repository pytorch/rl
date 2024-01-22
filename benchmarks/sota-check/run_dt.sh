#!/bin/bash

#SBATCH --job-name=dt
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=dt_offline_output_%j.txt
#SBATCH --error=dt_offline_error_%j.txt

current_commit=$(git rev-parse HEAD)
project_name="torchrl-example-check-$current_commit"
python ../../examples/decision_transformer/dt.py \
  logger.backend=wandb \
  logger.project_name="$project_name" \
  logger.group_name="dt_offline"
