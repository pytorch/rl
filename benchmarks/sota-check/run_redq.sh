#!/bin/bash

#SBATCH --job-name=redq
#SBATCH --partition=test
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=redq_output_%j.txt
#SBATCH --error=redq_error_%j.txt

CUDA_VISIBLE_DEVICES=1 python ../../examples/redq/redq.py \
  num_workers=2 \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  collector.frames_per_batch=16 \
  collector.env_per_collector=1 \
  buffer.batch_size=10 \
  collector.device=cuda:0 \
  optim.steps_per_batch=1 \
  logger.record_video=True \
  logger.record_frames=4 \
  buffer.size=120 \
  logger.backend=wandb \
  logger.backend.project_name="sota-check"