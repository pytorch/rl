#!/bin/bash

# Function to display script usage
display_usage() {
    cat <<EOF
Usage: ./submitit-release-check.sh [OPTIONS]

OPTIONS:
  --partition PARTITION   Specify the Slurm partition for the job.
  --n_runs N_RUNS         Specify the number of runs for each script. Default is 1.

EXAMPLES:
  ./submitit-release-check.sh --partition <PARTITION_NAME> --n_runs 5

EOF
    return 1
}

# Check if the script is called with --help or without any arguments
if [ "$1" == "--help" ]; then
    display_usage
fi

# Initialize variables with default values
n_runs="1"
slurm_partition=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --n_runs)
      n_runs="$2"
      shift 2
      ;;
    --partition)
      slurm_partition="$2"
      shift 2
      ;;
    *)
      echo "$1 is not a valid argument. See './submitit-release-check.sh --help'."
      return 0
      ;;
  esac
done

scripts=(
    run_a2c_atari.sh
    run_a2c_mujoco.sh
    run_cql_offline.sh
    run_cql_online.sh
    run_ddpg.sh
    run_discrete_sac.sh
    run_dqn_atari.sh
    run_dqn_cartpole.sh
    run_impala_single_node.sh
    run_iql_offline.sh
    run_iql_online.sh
    run_iql_discrete.sh
    run_multiagent_iddpg.sh
    run_multiagent_ippo.sh
    run_multiagent_iql.sh
    run_multiagent_qmix.sh
    run_multiagent_sac.sh
    run_ppo_atari.sh
    run_ppo_mujoco.sh
    run_sac.sh
    run_td3.sh
    run_dt.sh
    run_dt_online.sh
)

mkdir -p "slurm_errors"
mkdir -p "slurm_logs"

# remove the previous report
rm -f report.log

# Submit jobs with the specified partition the specified number of times
if [ -z "$slurm_partition" ]; then
    for script in "${scripts[@]}"; do
        for ((i=1; i<=$n_runs; i++)); do
            sbatch "$script"
        done
    done
else
  for script in "${scripts[@]}"; do
      for ((i=1; i<=$n_runs; i++)); do
          sbatch --partition="$slurm_partition" "$script"
      done
  done
fi
