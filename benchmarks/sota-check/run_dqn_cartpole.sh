#!/bin/bash

#SBATCH --job-name=dqn_cartpole
#SBATCH --partition=test
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=dqn_cartpole_output_%j.txt
#SBATCH --error=dqn_cartpole_error_%j.txt

CUDA_VISIBLE_DEVICES=1 python ../../examples/dqn/dqn_cartpole.py \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  collector.frames_per_batch=16 \
  buffer.batch_size=10 \
  device=cuda:0 \
  loss.num_updates=1 \
  buffer.buffer_size=120 \
  logger.backend=wandb \
  logger.backend.project_name="sota-check"