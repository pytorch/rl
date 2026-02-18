#!/bin/bash

#SBATCH --job-name=discrete_sac
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=discrete_sac_output_%j.txt
#SBATCH --error=discrete_sac_error_%j.txt

python ../../examples/discrete_sac/discrete_sac.py
