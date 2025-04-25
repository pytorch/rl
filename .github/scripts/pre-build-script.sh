#!/bin/bash

pip install --upgrade setuptools

${CONDA_RUN} pip install "pybind11[global]"
${CONDA_RUN} conda install anaconda::cmake -y
${CONDA_RUN} pip install git+https://github.com/pytorch/tensordict.git -U
