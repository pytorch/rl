#!/bin/bash

#SBATCH --job-name=discrete_sac
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/discrete_sac_%j.txt
#SBATCH --error=slurm_errors/discrete_sac_%j.txt

current_commit=$(git rev-parse --short HEAD)
project_name="torchrl-example-check-$current_commit"
group_name="discrete_sac"
export PYTHONPATH=$(dirname $(dirname $PWD))
python $PYTHONPATH/sota-implementations/discrete_sac/discrete_sac.py \
  logger.backend=wandb \
  logger.project_name="$project_name" \
  logger.group_name="$group_name"

# Capture the exit status of the Python command
exit_status=$?
# Write the exit status to a file
if [ $exit_status -eq 0 ]; then
  echo "${group_name}_${SLURM_JOB_ID}=success" >> report.log
else
  echo "${group_name}_${SLURM_JOB_ID}=error" >> report.log
fi
