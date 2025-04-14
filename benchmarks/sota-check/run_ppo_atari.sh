#!/bin/bash

#SBATCH --job-name=ppo_atari
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=ppo_atari_output_%j.txt
#SBATCH --error=ppo_atari_error_%j.txt

python ../../examples/ppo/ppo_atari.py
