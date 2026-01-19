#!/bin/bash

pip install --upgrade setuptools
${CONDA_RUN} pip install "pybind11[global]"
${CONDA_RUN} pip install git+https://github.com/pytorch/tensordict.git -U --no-deps

export TORCHRL_BUILD_VERSION=0.10.0
