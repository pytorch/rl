# SOTA Performance checks

This folder contains a `submitit-release-check.sh` file that executes all
the training scripts using `sbatch` with the default configuration and long them
into a common WandB project.

This script is to be executed before every release to assess the performance of
the various algorithms available in torchrl. The name of the project will include
the specific commit of torchrl used to run the scripts (e.g. `torchrl-examples-check-<commit>`).

## Usage

To display the script usage, you can use the `--help` option:

```bash
./submitit-release-check.sh --help
