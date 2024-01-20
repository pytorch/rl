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
  logger.backend=wandb \
  logger.project_name="$project_name" \
  logger.group_name="td3"
