#!/bin/bash

#SBATCH --job-name=marl_iddpg
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=marl_iddpg_output_%j.txt
#SBATCH --error=marl_iddpg_error_%j.txt

python ../../examples/multiagent/maddpg_iddpg.py
