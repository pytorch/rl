#!/usr/bin/env bash

unset PYTORCH_VERSION
# For unittest, nightly PyTorch is used as the following section,
# so no need to set PYTORCH_VERSION.
# In fact, keeping PYTORCH_VERSION forces us to hardcode PyTorch version in config.

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

# suggested by habitat team: do we need this?
#conda create -n $MY_TEST_ENV python=3.7 cmake=3.14.0 -y
#conda activate $MY_TEST_ENV

# If you already have an environment you want to use, you can just run the following:
conda install habitat-sim withbullet headless -c conda-forge -c aihabitat-nightly -y
pip3 install gym
pip3 install git+https://github.com/facebookresearch/habitat-lab.git#subdirectory=habitat-lab

# This is to reduce verbosity
export MAGNUM_LOG=quiet && export HABITAT_SIM_LOG=quiet
