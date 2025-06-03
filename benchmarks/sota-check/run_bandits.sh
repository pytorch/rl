#!/bin/bash

#SBATCH --job-name=bandits_dqn
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=bandits_dqn_output_%j.txt
#SBATCH --error=bandits_dqn_error_%j.txt

python ../../examples/bandits/dqn.py
