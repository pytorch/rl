#!/bin/bash

#SBATCH --job-name=dqn_atari
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=dqn_atari_output_%j.txt
#SBATCH --error=dqn_atari_error_%j.txt

current_commit=$(git rev-parse HEAD)
project_name="torchrl-example-check-$current_commit"
python ../../examples/dqn/dqn_atari.py \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  collector.frames_per_batch=16 \
  buffer.batch_size=10 \
  device=cuda:0 \
  loss.num_updates=1 \
  buffer.buffer_size=120 \
  logger.backend=wandb \
  logger.project_name="$project_name" \
  logger.group_name="dqn_atari"
