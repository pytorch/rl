#!/bin/bash

#SBATCH --job-name=marl_sac
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=marl_sac_output_%j.txt
#SBATCH --error=marl_sac_error_%j.txt

current_commit=$(git rev-parse HEAD)
project_name="sota-check_$current_commit"
python ../../examples/multiagent/sac.py \
  collector.n_iters=2 \
  collector.frames_per_batch=200 \
  train.num_epochs=3 \
  train.minibatch_size=100 \
  logger.backend=wandb \
  logger.project_name="$project_name"
