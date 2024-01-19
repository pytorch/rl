#!/bin/bash

#SBATCH --job-name=rlhf
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=rlhf_output_%j.txt
#SBATCH --error=rlhf_error_%j.txt

current_commit=$(git rev-parse HEAD)
project_name="torchrl-example-check-$current_commit"
python ../../examples/rlhf/train_rlhf.py \
  sys.device=cuda:0 sys.ref_device=cuda:0 \
  model.name_or_path=gpt2 train.max_epochs=2 \
  data.batch_size=2 train.ppo.ppo_batch_size=2 \
  train.ppo.ppo_num_epochs=1 reward_model.name_or_path= \
  train.ppo.episode_length=8 train.ppo.num_rollouts_per_epoch=4 \
  data.block_size=110 \
  io.logger=wandb \
  io.project_name="$project_name" \
  io.group_name="rlhf"
