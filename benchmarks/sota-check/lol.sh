#!/bin/bash

# Function to display script usage
display_usage() {
    echo "Usage: ./submitit-release-check.sh [--partition PARTITION] [--n_runs N_RUNS]"
    echo "  --partition: (Optional) Specify the Slurm partition for the job."
    echo "  PARTITION: Specify the Slurm partition if --partition is used. "
    echo "  --n_runs: (Optional) Specify the number of runs for each script. Default is 1."
    return 1
}

# Check if the script is called with --help or without any arguments
if [ "$1" == "--help" ]; then
    display_usage
fi

