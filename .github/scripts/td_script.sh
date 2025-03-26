#!/bin/bash

export TORCHRL_BUILD_VERSION=0.7.0
pip install --upgrade setuptools

if pip list | grep -q torch; then
    echo "Torch is installed."
    ${CONDA_RUN} pip install git+https://github.com/pytorch/tensordict.git -U
else
    echo "Torch is not installed - tensordict will be installed later."
fi
