#!/bin/bash

export TORCHRL_BUILD_VERSION=0.10.0
${CONDA_RUN} pip install --upgrade setuptools

# Always install pybind11 - required for building C++ extensions
${CONDA_RUN} pip install "pybind11[global]"

# Check if ARCH is set to aarch64
ARCH=${ARCH:-}  # This sets ARCH to an empty string if it's not defined

if ${CONDA_RUN} pip list | grep -q torch; then
    echo "Torch is installed."
    ${CONDA_RUN} pip install git+https://github.com/pytorch/tensordict.git -U --no-deps
elif [[ -n "${SMOKE_TEST_SCRIPT:-}" ]]; then
    ${CONDA_RUN} ${PIP_INSTALL_TORCH}
    ${CONDA_RUN} pip install git+https://github.com/pytorch/tensordict.git -U --no-deps
else
    echo "Torch is not installed - tensordict will be installed later."
fi
