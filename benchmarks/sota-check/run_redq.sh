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
  logger.backend=wandb \
  logger.project_name="$project_name" \
  logger.group_name="redq"
