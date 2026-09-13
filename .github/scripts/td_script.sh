#!/bin/bash

# test-infra maps release branches to its test index. Override its install
# command before the first install so release wheels use stable PyTorch.
if [[ "${GITHUB_REF_NAME:-}" == release/* || ("${GITHUB_REF_TYPE:-}" == "tag" && "${GITHUB_REF_NAME:-}" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$) ]]; then
    export TORCHRL_RELEASE_TORCH_VERSION=2.14.0
    export PIP_INSTALL_TORCH="pip install torch==${TORCHRL_RELEASE_TORCH_VERSION} --index-url https://download.pytorch.org/whl/${CU_VERSION:-cpu}"
fi

export TORCHRL_BUILD_VERSION="${BUILD_VERSION:-0.14.1}"
# PyPI rejects local versions such as X.Y.Z+cpu. CPU wheels are the
# default PyPI artifacts, so strip only the CPU build suffix while keeping
# CUDA/ROCm suffixes for the extra-index wheels.
if [[ "${TORCHRL_BUILD_VERSION}" == *+cpu ]]; then
    export TORCHRL_BUILD_VERSION="${TORCHRL_BUILD_VERSION%+cpu}"
fi
${CONDA_RUN} pip install --upgrade setuptools packaging

# Always install pybind11 - required for building C++ extensions
${CONDA_RUN} pip install "pybind11[global]"
${CONDA_RUN} pip install cloudpickle importlib_metadata numpy orjson "pyvers>=0.2.3,<0.3.0"

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

if ${CONDA_RUN} pip show torch >/dev/null 2>&1; then
    echo "Torch is installed."
    # This script is sourced again before compilation and wheel smoke tests.
    if [[ -n "${TORCHRL_RELEASE_TORCH_VERSION:-}" ]]; then
        ${CONDA_RUN} python -c 'import os, torch; expected = os.environ["TORCHRL_RELEASE_TORCH_VERSION"]; assert torch.__version__.split("+", 1)[0] == expected, f"Expected stable PyTorch {expected}, got {torch.__version__}"; print(f"Release build PyTorch: {torch.__version__}")'
    fi
    install_tensordict
elif [[ -n "${SMOKE_TEST_SCRIPT:-}" ]]; then
    ${CONDA_RUN} ${PIP_INSTALL_TORCH}
    install_tensordict
else
    echo "Torch is not installed - tensordict will be installed later."
fi
