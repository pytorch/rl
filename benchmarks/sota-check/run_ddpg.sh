#!/bin/bash

#SBATCH --job-name=ddpg
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=ddpg_output_%j.txt
#SBATCH --error=ddpg_error_%j.txt

current_commit=$(git rev-parse HEAD)
project_name="sota-check_$current_commit"
python ../../examples/ddpg/ddpg.py \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  optim.batch_size=10 \
  collector.frames_per_batch=16 \
  collector.env_per_collector=1 \
  collector.device=cuda:0 \
  network.device=cuda:0 \
  optim.utd_ratio=1 \
  replay_buffer.size=120 \
  env.name=Pendulum-v1 \
  logger.backend=wandb \
  logger.project_name="$project_name"
