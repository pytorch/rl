#!/usr/bin/env bash

# This script is for setting up environment in which unit test is ran.
# To speed up the CI time, the resulting environment is cached.
#
# Do not install PyTorch and torchvision here, otherwise they also get cached.

set -e

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'
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

# 4. Install StarCraft 2 with SMAC maps
# SC2PATH is set in run_test.sh
printf "* Installing StarCraft 2 and SMAC maps into '${root_dir}/smac/StarCraftII'\n"
mkdir $root_dir/smac
cd $root_dir/smac/
# TODO: discuss how we can cache it to avoid downloading ~4 GB on each run.
wget https://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
# The archive contains StarCraftII folder. Password comes from the documentation.
unzip -qo -P iagreetotheeula SC2.4.10.zip
wget https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip
tar -xf SMAC_Maps.zip --directory ./StarCraftII/Maps
