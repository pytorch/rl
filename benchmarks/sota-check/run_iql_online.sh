#!/bin/bash

#SBATCH --job-name=iql_online
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/iql_online_output_%j.txt
#SBATCH --error=slurm_errors/iql_online_error_%j.txt

current_commit=$(git rev-parse --short HEAD)
project_name="torchrl-example-check-$current_commit"
group_name="iql_online"
export PYTHONPATH=$(dirname $(dirname $PWD))
python $PYTHONPATH/examples/iql/iql_online.py \
  logger.backend=wandb \
  logger.project_name="$project_name" \
  logger.group_name="iql_online"

# Capture the exit status of the Python command
exit_status=$?
# Write the exit status to a file
if [ $exit_status -eq 0 ]; then
  echo "${group_name}_${SLURM_JOB_ID}=success" >> report.log
else
  echo "${group_name}_${SLURM_JOB_ID}=error" >> report.log
fi
