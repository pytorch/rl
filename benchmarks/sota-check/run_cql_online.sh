#!/bin/bash

#SBATCH --job-name=cql_online
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=cql_online_output_%j.txt
#SBATCH --error=cql_online_error_%j.txt

current_commit=$(git rev-parse HEAD)
project_name="torchrl-example-check-$current_commit"
python ../../examples/cql/cql_online.py \
  logger.backend=wandb \
  logger.project_name="$project_name" \
  logger.group_name="cql_online"
