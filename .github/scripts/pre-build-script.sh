#!/bin/bash

pip install --upgrade setuptools

${CONDA_RUN} pip install git+https://github.com/pytorch/tensordict.git -U
