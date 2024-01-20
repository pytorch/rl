#!/bin/bash

#SBATCH --job-name=ppo_mujoco
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=ppo_mujoco_output_%j.txt
#SBATCH --error=ppo_mujoco_error_%j.txt

current_commit=$(git rev-parse HEAD)
project_name="torchrl-example-check-$current_commit"
python ../../examples/ppo/ppo_mujoco.py  \
  logger.backend=wandb \
  logger.project_name="$project_name" \
  logger.group_name="ppo_mujoco"
