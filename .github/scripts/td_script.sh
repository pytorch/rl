#!/bin/bash

export TORCHRL_BUILD_VERSION=0.11.0
${CONDA_RUN} pip install --upgrade setuptools

# Always install pybind11 - required for building C++ extensions
${CONDA_RUN} pip install "pybind11[global]"

# Check if ARCH is set to aarch64
ARCH=${ARCH:-}  # This sets ARCH to an empty string if it's not defined

# Determine tensordict installation source based on branch/tag
# - release/* branches or release tags: use PyPI stable release
# - main, PRs, nightly, etc.: use git (latest development version)
install_tensordict() {
    local source="${TENSORDICT_SOURCE:-auto}"
    
    if [[ "$source" == "stable" ]]; then
        echo "Installing tensordict from PyPI (stable) - explicit override"
        ${CONDA_RUN} pip install tensordict -U --no-deps
    elif [[ "$source" == "git" ]]; then
        echo "Installing tensordict from git - explicit override"
        ${CONDA_RUN} pip install git+https://github.com/pytorch/tensordict.git -U --no-deps
    elif [[ "$GITHUB_REF_TYPE" == "branch" && "$GITHUB_REF_NAME" == release/* ]]; then
        echo "Installing tensordict from PyPI (stable) - detected release branch: $GITHUB_REF_NAME"
        ${CONDA_RUN} pip install tensordict -U --no-deps
    elif [[ "$GITHUB_REF_TYPE" == "tag" && "$GITHUB_REF_NAME" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "Installing tensordict from PyPI (stable) - detected release tag: $GITHUB_REF_NAME"
        ${CONDA_RUN} pip install tensordict -U --no-deps
    else
        echo "Installing tensordict from git - branch: ${GITHUB_REF_NAME:-unknown}, type: ${GITHUB_REF_TYPE:-unknown}"
        ${CONDA_RUN} pip install git+https://github.com/pytorch/tensordict.git -U --no-deps
    fi
}

if ${CONDA_RUN} pip list | grep -q torch; then
    echo "Torch is installed."
    install_tensordict
elif [[ -n "${SMOKE_TEST_SCRIPT:-}" ]]; then
    ${CONDA_RUN} ${PIP_INSTALL_TORCH}
    install_tensordict
else
    echo "Torch is not installed - tensordict will be installed later."
fi
