#!/bin/bash

export BUILD_VERSION=0.3.2

${CONDA_RUN} pip install git+https://github.com/pytorch/tensordict.git -U
