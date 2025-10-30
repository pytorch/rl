#!/bin/bash

export TORCHRL_BUILD_VERSION=0.10.1
pip install --upgrade setuptools

# Check if ARCH is set to aarch64
ARCH=${ARCH:-}  # This sets ARCH to an empty string if it's not defined

if pip list | grep -q torch; then
    echo "Torch is installed."

    # ${CONDA_RUN} conda install 'anaconda::cmake>=3.22' -y

    ${CONDA_RUN} pip install "pybind11[global]"

    ${CONDA_RUN} pip install git+https://github.com/pytorch/tensordict.git -U --no-deps
elif [[ -n "${SMOKE_TEST_SCRIPT:-}" ]]; then
    ${CONDA_RUN} ${PIP_INSTALL_TORCH}
    #    TODO: revert when nightlies of tensordict are fixed
    #    if [[ "$ARCH" == "aarch64" ]]; then


#     ${CONDA_RUN} conda install 'anaconda::cmake>=3.22' -y

     ${CONDA_RUN} pip install "pybind11[global]"

    ${CONDA_RUN} pip install git+https://github.com/pytorch/tensordict.git -U --no-deps
else
    echo "Torch is not installed - tensordict will be installed later."
fi
