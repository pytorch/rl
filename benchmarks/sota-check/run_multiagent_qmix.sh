#!/bin/bash

#SBATCH --job-name=marl_qmix_vdn
#SBATCH --partition=test
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=marl_qmix_vdn_output_%j.txt
#SBATCH --error=marl_qmix_vdn_error_%j.txt

python ../../examples/multiagent/qmix_vdn.py \
  collector.n_iters=2 \
  collector.frames_per_batch=200 \
  train.num_epochs=3 \
  train.minibatch_size=100 \
  logger.backend=wandb \
  logger.project_name="sota-check"

