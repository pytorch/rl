#!/bin/bash

#SBATCH --job-name=ddpg
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=ddpg_output_%j.txt
#SBATCH --error=ddpg_error_%j.txt

python ../../examples/ddpg/ddpg.py
