#!/bin/bash

#SBATCH --job-name=ddpg
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=ddpg_output_%j.txt
#SBATCH --error=ddpg_error_%j.txt

current_commit=$(git rev-parse HEAD)
project_name="torchrl-example-check-$current_commit"
python ../../examples/ddpg/ddpg.py \
  logger.backend=wandb \
  logger.project_name="$project_name" \
  logger.group_name="ddpg"
