#!/bin/bash

#SBATCH --job-name=cql_offline
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=cql_offline_output_%j.txt
#SBATCH --error=cql_offline_error_%j.txt

python ../../examples/cql/cql_offline.py
