#!/bin/bash

#SBATCH --job-name=ppo_mujoco
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=ppo_mujoco_output_%j.txt
#SBATCH --error=ppo_mujoco_error_%j.txt

python ../../examples/ppo/ppo_mujoco.py
