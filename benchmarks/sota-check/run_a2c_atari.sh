#!/bin/bash

#SBATCH --job-name=a2c_atari
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=a2c_atari_output_%j.txt
#SBATCH --error=a2c_atari_error_%j.txt

CUDA_VISIBLE_DEVICES=1 python ../../examples/a2c/a2c_atari.py \
  collector.total_frames=80 \
  collector.frames_per_batch=20 \
  loss.mini_batch_size=20 \
  logger.backend=wandb \
  logger.backend_kwargs="{"wandb_kwargs": {"project": "sota-check"}}" \
  logger.test_interval=40
