#!/bin/bash

#SBATCH --job-name=marl_sac
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=marl_sac_output_%j.txt
#SBATCH --error=marl_sac_error_%j.txt

python ../../examples/multiagent/sac.py
