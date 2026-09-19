# SOTA Performance checks

This folder contains a `submitit-release-check.sh` file that executed all
the training scripts using `sbatch` with the default configuration and a wandb
logger.

This script is to be executed before every release to assess the performance of
the various algorithms available in torchrl.
