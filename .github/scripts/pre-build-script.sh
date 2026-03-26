#!/bin/bash

${CONDA_RUN} pip install --upgrade setuptools

${CONDA_RUN} pip install "pybind11[global]"
${CONDA_RUN} conda install anaconda::cmake -y

# Determine tensordict installation source based on branch/tag
# - release/* branches or release tags: use PyPI stable release
# - main, PRs, nightly, etc.: use git (latest development version)
install_tensordict() {
    local source="${TENSORDICT_SOURCE:-auto}"
    
    if [[ "$source" == "stable" ]]; then
        echo "Installing tensordict from PyPI (stable) - explicit override"
        ${CONDA_RUN} pip install tensordict -U
    elif [[ "$source" == "git" ]]; then
        echo "Installing tensordict from git - explicit override"
        ${CONDA_RUN} pip install git+https://github.com/pytorch/tensordict.git -U
    elif [[ "$GITHUB_REF_TYPE" == "branch" && "$GITHUB_REF_NAME" == release/* ]]; then
        echo "Installing tensordict from PyPI (stable) - detected release branch: $GITHUB_REF_NAME"
        ${CONDA_RUN} pip install tensordict -U
    elif [[ "$GITHUB_REF_TYPE" == "tag" && "$GITHUB_REF_NAME" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "Installing tensordict from PyPI (stable) - detected release tag: $GITHUB_REF_NAME"
        ${CONDA_RUN} pip install tensordict -U
    else
        echo "Installing tensordict from git - branch: ${GITHUB_REF_NAME:-unknown}, type: ${GITHUB_REF_TYPE:-unknown}"
        ${CONDA_RUN} pip install git+https://github.com/pytorch/tensordict.git -U
    fi
}

install_tensordict
