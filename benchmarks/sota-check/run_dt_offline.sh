#!/bin/bash

#SBATCH --job-name=dt_offline
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=dt_offline_output_%j.txt
#SBATCH --error=dt_offline_error_%j.txt

python ../../examples/dt/dt_offline.py
