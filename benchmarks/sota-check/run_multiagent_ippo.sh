#!/bin/bash

#SBATCH --job-name=marl_ippo
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=marl_ippo_output_%j.txt
#SBATCH --error=marl_ippo_error_%j.txt

python ../../examples/multiagent/maddpg_ippo.py
