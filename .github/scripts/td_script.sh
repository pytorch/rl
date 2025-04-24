#!/bin/bash

export TORCHRL_BUILD_VERSION=0.8.0
pip install --upgrade setuptools

# Check if ARCH is set to aarch64
ARCH=${ARCH:-}  # This sets ARCH to an empty string if it's not defined

if pip list | grep -q torch; then
    echo "Torch is installed."
    if [[ "$ARCH" == "aarch64" ]]; then
        ${CONDA_RUN} pip install git+https://github.com/pytorch/tensordict.git -U --no-deps
    else
        ${CONDA_RUN} pip install tensordict-nightly -U
    fi
elif [[ -n "${SMOKE_TEST_SCRIPT:-}" ]]; then
    ${CONDA_RUN} ${PIP_INSTALL_TORCH}
    if [[ "$ARCH" == "aarch64" ]]; then
        ${CONDA_RUN} pip install git+https://github.com/pytorch/tensordict.git -U --no-deps
    else
        ${CONDA_RUN} pip install tensordict-nightly -U
    fi
else
    echo "Torch is not installed - tensordict will be installed later."
fi
