#!/bin/bash

#SBATCH --job-name=marl_iql
#SBATCH --partition=test
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=marl_iql_output_%j.txt
#SBATCH --error=marl_iql_error_%j.txt

python ../../examples/multiagent/iql.py \
  collector.n_iters=2 \
  collector.frames_per_batch=200 \
  train.num_epochs=3 \
  train.minibatch_size=100 \
  logger.backend=wandb \
  logger.project_name="sota-check"
