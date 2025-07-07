#!/bin/bash

set -euo pipefail

# Define cleanup function BEFORE trap
cleanup() {
    if command -v ray &>/dev/null; then
        echo "Stopping Ray on node $CURRENT_NODE"
        ray stop || true
    fi
}

# Set up cleanup trap early
trap cleanup EXIT

# Utility to check required environment variables
check_env_var() {
    if [ -z "${!1}" ]; then
        echo "Error: Required environment variable $1 is not set"
        exit 1
    fi
}

CURRENT_NODE=$(hostname | cut -d. -f1)
CMD="$1"

echo "SLURM_NODEID: $SLURM_NODEID"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "Current node: $CURRENT_NODE"
echo "Head node: $HEAD_NODE ($HEAD_NODE_IP)"
echo "Ray port: $RAY_PORT"
echo "Command: $CMD"

check_env_var "HEAD_NODE"
check_env_var "HEAD_NODE_IP"
check_env_var "RAY_PORT"
check_env_var "SLURM_NODEID"
check_env_var "SLURM_NNODES"

# Node 0 is the Ray head node
if [ "$SLURM_NODEID" -eq 0 ]; then
    echo "Starting Ray head on Node 0"
    ray start --head --disable-usage-stats --port="$RAY_PORT"
    echo "Ray head node started at $HEAD_NODE_IP:$RAY_PORT"

    echo "Ray head is running on $CURRENT_NODE — waiting indefinitely to keep cluster alive..."
    sleep infinity
else
    echo "Waiting for Ray head node to be ready..."
    sleep 10

    echo "Starting Ray worker on node $CURRENT_NODE (ID: $SLURM_NODEID)"
    ray start --disable-usage-stats --address="$HEAD_NODE_IP:$RAY_PORT" || {
        echo "Failed to start Ray worker"
        exit 1
    }

    echo "Running command on worker node $CURRENT_NODE"
    bash -c "$CMD"
fi

echo "Node $CURRENT_NODE: Done"
