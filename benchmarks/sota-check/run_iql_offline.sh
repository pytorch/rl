#!/bin/bash

#SBATCH --job-name=iql_offline
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=iql_offline_output_%j.txt
#SBATCH --error=iql_offline_error_%j.txt

python ../../examples/iql/iql_offline.py
