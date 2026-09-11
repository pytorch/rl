#!/bin/bash

#SBATCH --job-name=cql_online
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=cql_online_output_%j.txt
#SBATCH --error=cql_online_error_%j.txt

python ../../examples/cql/cql_online.py
