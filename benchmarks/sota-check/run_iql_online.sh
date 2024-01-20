#!/bin/bash

#SBATCH --job-name=iql_online
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=iql_online_output_%j.txt
#SBATCH --error=iql_online_error_%j.txt

current_commit=$(git rev-parse HEAD)
project_name="torchrl-example-check-$current_commit"
python ../../examples/iql/iql_online.py \
  logger.backend=wandb \
  logger.project_name="$project_name" \
  logger.group_name="iql_online"
