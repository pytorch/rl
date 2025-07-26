#!/bin/bash

set -euo pipefail

# Get command from argument
CMD="$1"

# Set up Ray cluster configuration
HEAD_NODE=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
RAY_PORT=6379

# Get current node name 
CURRENT_NODE=$(hostname | cut -d. -f1)

# Get HEAD_NODE_IP
if [ "$SLURM_NODEID" -eq 0 ]; then
    # We're on the head node, get our own IP
    HEAD_NODE_IP=$(hostname -I | awk '{print $1}')
else
    # We're on a worker, resolve the head node's IP using DNS
    HEAD_NODE_IP=$(getent hosts "$HEAD_NODE" | awk '{print $1}')
fi

# Set up cleanup function
cleanup() {
    if command -v ray &>/dev/null; then
        echo "Stopping Ray on node $CURRENT_NODE"
        ray stop || true
    fi
}
trap cleanup EXIT

# Start Ray based on node role
if [ "$SLURM_NODEID" -eq 0 ]; then
    echo "Starting Ray head node on $CURRENT_NODE"
    ray start --head --disable-usage-stats --port=$RAY_PORT
    echo "Ray head node started at $HEAD_NODE_IP:$RAY_PORT"
else
    echo "Waiting for head node to be ready..."
    sleep 10
    echo "Starting Ray worker on node $CURRENT_NODE (ID: $SLURM_NODEID)"
    ray start --disable-usage-stats --address="$HEAD_NODE_IP:$RAY_PORT"
fi

# Ensure Ray cluster is ready
sleep 2

# Only head node runs the training command
if [ "$SLURM_NODEID" -eq 0 ]; then
    echo "Starting training process on head node $CURRENT_NODE"
    bash -c "$CMD"
else
    # Worker nodes just wait for the head to finish
    while ray status --address="$HEAD_NODE_IP:$RAY_PORT" &>/dev/null; do
        sleep 10
    done
fi

echo "Node $CURRENT_NODE: Done"
