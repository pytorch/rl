#!/usr/bin/env bash

unset PYTORCH_VERSION
# For unittest, nightly PyTorch is used as the following section,
# so no need to set PYTORCH_VERSION.
# In fact, keeping PYTORCH_VERSION forces us to hardcode PyTorch version in config.

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

conda install -c anaconda mesa-libgl-cos6-x86_64 -y
conda install -c anaconda mesa-libegl-cos6-x86_64 -y

# suggested by habitat team: do we need this?
#conda create -n $MY_TEST_ENV python=3.7 cmake=3.14.0 -y
#conda activate $MY_TEST_ENV

# If you already have an environment you want to use, you can just run the following:
conda install habitat-sim withbullet headless -c conda-forge -c aihabitat-nightly -y
conda run python -m pip install install git+https://github.com/facebookresearch/habitat-lab.git#subdirectory=habitat-lab
conda run python -m pip install install gym[accept-rom-license]
conda run python -m pip install install gym[atari]

# This is to reduce verbosity
conda env config vars set MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet \
  LD_PRELOAD="/home/circleci/project/conda/pkgs/mesa-libegl-cos6-x86_64-11.0.7-4/x86_64-conda_cos6-linux-gnu/sysroot/usr/lib64/libEGL.so.1 /home/circleci/project/conda/pkgs/mesa-libgl-cos6-x86_64-11.0.7-4/x86_64-conda_cos6-linux-gnu/sysroot/usr/lib64/libGL.so.1"

conda deactivate
conda activate ./env

# smoke test
python -c "import habitat;import habitat.utils.gym_definitions"
