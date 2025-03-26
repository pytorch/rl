#!/bin/bash

pip install --upgrade setuptools==72.1.0

export TORCHRL_BUILD_VERSION=0.7.0

${CONDA_RUN} pip install git+https://github.com/pytorch/tensordict.git -U
