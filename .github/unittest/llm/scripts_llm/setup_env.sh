#!/usr/bin/env bash

# This script is for setting up environment in which unit test is ran.
# To speed up the CI time, the resulting environment is cached.
#
# Do not install PyTorch and torchvision here, otherwise they also get cached.

set -e
export DEBIAN_FRONTEND=noninteractive
export TZ=UTC
apt-get update
apt-get install -yq --no-install-recommends git cmake
# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'
apt-get install -yq --no-install-recommends wget \
    gcc \
    g++ \
    unzip \
    curl \
    patchelf \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    swig3.0 \
    libglew-dev \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2

# Upgrade specific package
apt-get install -yq --no-install-recommends --only-upgrade libstdc++6

apt-get clean
rm -rf /var/lib/apt/lists/*

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
root_dir="$(git rev-parse --show-toplevel)"
conda_dir="${root_dir}/conda"
env_dir="${root_dir}/env"

cd "${root_dir}"

case "$(uname -s)" in
    Darwin*) os=MacOSX;;
    *) os=Linux
esac

# 1. Install conda at ./conda
if [ ! -d "${conda_dir}" ]; then
    printf "* Installing conda\n"
    wget -O miniconda.sh "http://repo.continuum.io/miniconda/Miniconda3-latest-${os}-x86_64.sh"
    bash ./miniconda.sh -b -f -p "${conda_dir}"
fi
eval "$(${conda_dir}/bin/conda shell.bash hook)"

# 2. Create test environment at ./env
printf "python: ${PYTHON_VERSION}\n"
if [ ! -d "${env_dir}" ]; then
    printf "* Creating a test environment\n"
    conda create --prefix "${env_dir}" -y python="$PYTHON_VERSION"
fi
conda activate "${env_dir}"


# 3. Install Conda dependencies
printf "* Installing dependencies (except PyTorch)\n"
echo "  - python=${PYTHON_VERSION}" >> "${this_dir}/environment.yml"
cat "${this_dir}/environment.yml"

pip install pip --upgrade

conda env update --file "${this_dir}/environment.yml" --prune
