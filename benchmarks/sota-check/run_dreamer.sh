#!/bin/bash

#SBATCH --job-name=dreamer
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=dreamer_output_%j.txt
#SBATCH --error=dreamer_error_%j.txt

python ../../examples/dreamer/dreamer.py
