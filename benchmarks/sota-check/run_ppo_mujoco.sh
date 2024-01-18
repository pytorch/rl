#!/bin/bash

#SBATCH --job-name=ppo_mujoco
#SBATCH --partition=test
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=ppo_mujoco_output_%j.txt
#SBATCH --error=ppo_mujoco_error_%j.txt

# Activate your virtual environment if needed
conda activate torch_rl2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/abou/.mujoco/mujoco210/bin

python ../../examples/ppo/ppo_mujoco.py

# Deactivate the virtual environment if activated
conda deactivate
