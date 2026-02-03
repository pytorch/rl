#!/bin/bash
set -euo pipefail

cd /root

apt-get update
apt-get install -y --no-install-recommends \
    git \
    cmake \
    ninja-build \
    build-essential

# Create venv only if it doesn't exist
if [ ! -d "/root/torchrl-llm" ]; then
    uv venv torchrl-llm --python 3.12
fi
source torchrl-llm/bin/activate

# Helper function to safely pull a branch, with fallback to delete/refetch/checkout
safe_git_pull() {
    local branch=$1
    local remote_branch=$2
    if ! git pull; then
        echo "Git pull failed, resetting branch..."
        git checkout HEAD~0  # detach HEAD
        git branch -D "$branch" || true
        git fetch origin
        git checkout -b "$branch" "origin/$remote_branch"
    fi
}

# Python build deps for local builds
uv pip install -U setuptools wheel
uv pip install "pybind11[global]" numpy

if [ ! -d "/root/tensordict" ]; then
    git clone --single-branch --branch main https://github.com/pytorch/tensordict
fi
cd /root/tensordict
git fetch origin
git checkout main
safe_git_pull "main" "main"
uv pip install --no-deps -e .

cd /root
if [ ! -d "/root/rl" ]; then
    git clone --single-branch --branch main https://github.com/pytorch/rl
fi
cd /root/rl
git fetch origin
git checkout main
safe_git_pull "main" "main"
uv pip install --no-deps -e .

# Latest vLLM from PyPI
uv pip install -U vllm

# Test dependencies
uv pip install pytest pytest-instafail pytest-timeout transformers ray

python -c "
import torch
import tensordict
import torchrl
import vllm

print('torch:', torch.__version__)
print('tensordict:', tensordict.__version__)
print('torchrl:', torchrl.__version__)
print('vllm:', vllm.__version__)
"
