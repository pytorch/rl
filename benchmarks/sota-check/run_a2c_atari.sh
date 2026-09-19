#!/bin/bash

#SBATCH --job-name=a2c_atari
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=a2c_atari_output_%j.txt
#SBATCH --error=a2c_atari_error_%j.txt

python ../../examples/a2c/a2c_atari.py
