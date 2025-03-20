#!/bin/bash

export TORCHRL_BUILD_VERSION=0.7.0

${CONDA_RUN} pip install git+https://github.com/pytorch/tensordict.git -U
