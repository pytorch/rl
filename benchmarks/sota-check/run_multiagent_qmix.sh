#!/bin/bash

#SBATCH --job-name=marl_qmix_vdn
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=marl_qmix_vdn_output_%j.txt
#SBATCH --error=marl_qmix_vdn_error_%j.txt

current_commit=$(git rev-parse HEAD)
project_name="torchrl-example-check-$current_commit"
python ../../examples/multiagent/qmix_vdn.py \
  logger.backend=wandb \
  logger.project_name="$project_name" \
  logger.group_name="marl_qmix_vdn"
