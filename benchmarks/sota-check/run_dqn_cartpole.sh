#!/bin/bash

#SBATCH --job-name=dqn_cartpole
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=dqn_cartpole_output_%j.txt
#SBATCH --error=dqn_cartpole_error_%j.txt

python ../../examples/dqn/dqn_cartpole.py
