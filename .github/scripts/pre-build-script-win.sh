#!/bin/bash

pip install --upgrade setuptools

export TORCHRL_BUILD_VERSION=0.7.0

${CONDA_RUN} pip install git+https://github.com/pytorch/tensordict.git -U
