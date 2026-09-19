#!/bin/bash

#SBATCH --job-name=dqn_atari
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=dqn_atari_output_%j.txt
#SBATCH --error=dqn_atari_error_%j.txt

python ../../examples/dqn/dqn_atari.py
