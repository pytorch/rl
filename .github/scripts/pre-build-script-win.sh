#!/bin/bash

pip install --upgrade setuptools

export TORCHRL_BUILD_VERSION=0.8.0

${CONDA_RUN} pip install "pybind11[global]"
${CONDA_RUN} conda install anaconda::cmake -y
${CONDA_RUN} pip install git+https://github.com/pytorch/tensordict.git -U
