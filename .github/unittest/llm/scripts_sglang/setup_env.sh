#!/usr/bin/env bash

# This script sets up the environment for SGLang tests using uv.
# SGLang has different Triton requirements than vLLM, so we need a separate environment.

set -e
export DEBIAN_FRONTEND=noninteractive
export TZ=UTC

apt-get update
apt-get install -yq --no-install-recommends git wget unzip curl patchelf
# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'

# Cleanup APT cache
apt-get clean && rm -rf /var/lib/apt/lists/*

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
root_dir="$(git rev-parse --show-toplevel)"
env_dir="${root_dir}/env"

cd "${root_dir}"

# Install uv if not present
if ! command -v uv &> /dev/null; then
    printf "* Installing uv\n"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create virtual environment using uv
printf "* Creating virtual environment with Python ${PYTHON_VERSION}\n"
if [ ! -d "${env_dir}" ]; then
    uv venv "${env_dir}" --python "${PYTHON_VERSION}"
fi

# Activate environment
source "${env_dir}/bin/activate"

printf "* Environment setup complete\n"
