#!/bin/bash

#SBATCH --job-name=a2c_mujoco
#SBATCH --partition=test
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=a2c_mujoco_output_%j.txt
#SBATCH --error=a2c_mujoco_error_%j.txt

python ../../examples/a2c/a2c_mujoco.py \
  env.env_name=HalfCheetah-v4 \
  collector.total_frames=40 \
  collector.frames_per_batch=20 \
  loss.mini_batch_size=10 \
  logger.backend=wandb \
  logger.backend_kwargs="{"wandb_kwargs": {"project": "sota-check"}}" \
  logger.test_interval=40
