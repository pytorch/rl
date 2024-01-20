#!/bin/bash

#SBATCH --job-name=iql_discrete
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=iql_discrete_output_%j.txt
#SBATCH --error=iql_discrete_error_%j.txt

current_commit=$(git rev-parse HEAD)
project_name="torchrl-example-check-$current_commit"
python ../../examples/iql/discrete_iql.py \
  logger.backend=wandb \
  logger.project_name="$project_name" \
  logger.group_name="iql_discrete"
