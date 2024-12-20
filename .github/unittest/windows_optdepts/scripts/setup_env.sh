#!/usr/bin/env bash

# This script is for setting up environment in which unit test is ran.
# To speed up the CI time, the resulting environment is cached.
#
# Do not install PyTorch and torchvision here, otherwise they also get cached.

set -e

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
root_dir="$(git rev-parse --show-toplevel)"
env_dir="${root_dir}/env"

cd "${root_dir}"

# 2. Create test environment at ./env
printf "* Creating a test environment\n"
conda create --prefix torchrl -y python="$PYTHON_VERSION"

printf "* Activating the environment"
conda deactivate
conda activate torchrl

printf "Python version"
echo $(which python)
echo $(python --version)
echo $(conda info -e)

#conda env update --file "${this_dir}/environment.yml" --prune

python -m pip install hypothesis future cloudpickle pytest pytest-cov pytest-mock pytest-instafail pytest-rerunfailures expecttest pyyaml scipy coverage
