#!/bin/bash

#SBATCH --job-name=transitive_rl
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/transitive_rl_%j.txt
#SBATCH --error=slurm_errors/transitive_rl_%j.txt

current_commit=$(git rev-parse --short HEAD)
project_name="torchrl-example-check-$current_commit"
group_name="transitive_rl"
export PYTHONPATH=$(dirname $(dirname $PWD))
python $PYTHONPATH/sota-implementations/transitive_rl/transitive_rl.py \
  logger.backend=wandb \
  logger.project="$project_name" \
  logger.group="$group_name" \
  compile.compile=true \
  compile.cudagraphs=true

exit_status=$?
if [ $exit_status -eq 0 ]; then
  echo "${group_name}_${SLURM_JOB_ID}=success" >> report.log
else
  echo "${group_name}_${SLURM_JOB_ID}=error" >> report.log
fi
