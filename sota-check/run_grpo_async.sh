#!/bin/bash

#SBATCH --job-name=grpo_async
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/grpo_async_%j.txt
#SBATCH --error=slurm_errors/grpo_async_%j.txt

current_commit=$(git rev-parse --short HEAD)
project_name="torchrl-example-check-$current_commit"
group_name="grpo_async"
export PYTHONPATH=$(dirname $(dirname $PWD))
python $PYTHONPATH/sota-implementations/grpo/grpo-async.py \
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
