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
conda run python -m pip install install git+https://github.com/facebookresearch/habitat-lab.git#subdirectory=habitat-lab
conda run python -m pip install install gym[accept-rom-license]
conda run python -m pip install install gym[atari]

# This is to reduce verbosity
conda env config vars set MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet

# smoke test
python -c "import habitat"
