#!/bin/bash

export TORCHRL_BUILD_VERSION=0.8.0
export NO_CPP_BINARIES=1

pip install git+https://github.com/pytorch/tensordict.git -U
