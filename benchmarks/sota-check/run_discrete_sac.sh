#!/bin/bash

#SBATCH --job-name=discrete_sac
#SBATCH --partition=test
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=discrete_sac_output_%j.txt
#SBATCH --error=discrete_sac_error_%j.txt

python ../../examples/discrete_sac/discrete_sac.py \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  collector.frames_per_batch=16 \
  collector.env_per_collector=1 \
  collector.device=cuda:0 \
  optim.batch_size=10 \
  optim.utd_ratio=1 \
  network.device=cuda:0 \
  optim.batch_size=10 \
  optim.utd_ratio=1 \
  replay_buffer.size=120 \
  env.name=CartPole-v1 \
  logger.backend=wandb \
  logger.project_name="sota-check"