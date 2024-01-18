#!/bin/bash

#SBATCH --job-name=ppo_atari
#SBATCH --partition=test
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=ppo_atari_output_%j.txt
#SBATCH --error=ppo_atari_error_%j.txt

CUDA_VISIBLE_DEVICES=1 python ../../examples/ppo/ppo_atari.py \
  collector.total_frames=80 \
  collector.frames_per_batch=20 \
  loss.mini_batch_size=20 \
  loss.ppo_epochs=2 \
  logger.backend=wandb \
  logger.project_name="sota-check" \
  logger.test_interval=10
