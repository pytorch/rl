#!/bin/bash

#SBATCH --job-name=sac
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=sac_output_%j.txt
#SBATCH --error=sac_error_%j.txt

python ../../examples/sac/sac.py
