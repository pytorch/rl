#!/bin/bash

# Function to display script usage
display_usage() {
    echo "Usage: ./submitit-release-check.sh [--partition PARTITION]"
    echo "  --partition: (Optional) Specify the Slurm partition for the job."
    echo "  PARTITION: Specify the Slurm partition if --partition is used."
    return 1
}

# Check if the script is called with --help or without any arguments
if [ "$1" == "--help" ]; then
    display_usage
fi

slurm_partition=""
# Check if the script is called with --partition and a subsequent argument
if [ "$1" == "--partition" ] && [ -n "$2" ]; then
    slurm_partition="$2"
    shift 2  # Consume the --partition option and its argument
fi

scripts=(
    run_a2c_atari.sh
    run_a2c_mujoco.sh
    run_cql_offline.sh
    run_cql_online.sh
    run_ddpg.sh
    run_discrete_sac.sh
    run_dqn_atari.sh
    run_dqn_cartpole.sh
    run_dt.sh
    run_dt_online.sh
    run_impala_single_node.sh
    run_iql_offline.sh
    run_iql_online.sh
    run_multiagent_iddpg.sh
    run_multiagent_ippo.sh
    run_multiagent_iql.sh
    run_multiagent_qmix.sh
    run_multiagent_sac.sh
    run_ppo_atari.sh
    run_ppo_mujoco.sh
    run_redq.sh
    run_sac.sh
    run_td3.sh
    #run_bandits.sh
    # run_rlhf.sh
)

# Submit jobs with the specified partition or empty string
if [ -z "$slurm_partition" ]; then
    sbatch "$script"
else
   sbatch --partition="$slurm_partition" "$script"
fi
