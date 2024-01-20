#!/bin/bash

#SBATCH --job-name=a2c_atari
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=a2c_atari_output_%j.txt
#SBATCH --error=a2c_atari_error_%j.txt

current_commit=$(git rev-parse HEAD)
project_name="torchrl-example-check-$current_commit"
python ../../examples/a2c/a2c_atari.py \
  logger.backend=wandb \
  logger.project_name="$project_name" \
  logger.group_name="a2c_atari"
