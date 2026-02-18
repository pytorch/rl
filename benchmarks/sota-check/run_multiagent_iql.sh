#!/bin/bash

#SBATCH --job-name=qmix_vdn
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=qmix_vdn_output_%j.txt
#SBATCH --error=qmix_vdn_error_%j.txt

python ../../examples/multiagent/qmix_vdn.py
