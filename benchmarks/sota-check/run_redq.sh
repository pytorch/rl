#!/bin/bash

#SBATCH --job-name=redq
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=redq_output_%j.txt
#SBATCH --error=redq_error_%j.txt

python ../../examples/redq/redq.py
