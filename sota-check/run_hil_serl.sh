#!/bin/bash

#SBATCH --job-name=hil_serl
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/hil_serl_%j.txt
#SBATCH --error=slurm_errors/hil_serl_%j.txt

current_commit=$(git rev-parse --short HEAD)
project_name="torchrl-example-check-$current_commit"
group_name="hil_serl"
export PYTHONPATH=$(dirname $(dirname $PWD))
export MUJOCO_GL=disabled OMP_NUM_THREADS=1
python $PYTHONPATH/sota-implementations/hil_serl/hil_serl.py \
  demos.path="${HIL_SERL_DEMOS:-null}" \
  logger.backend=wandb \
  logger.exp_name="$project_name"

# Capture the exit status of the Python command
exit_status=$?
# Write the exit status to a file
if [ $exit_status -eq 0 ]; then
  echo "${group_name}_${SLURM_JOB_ID}=success" >> report.log
else
  echo "${group_name}_${SLURM_JOB_ID}=error" >> report.log
fi
