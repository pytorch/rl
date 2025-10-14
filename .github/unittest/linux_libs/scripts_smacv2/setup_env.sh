#!/usr/bin/env bash

# This script is for setting up environment in which unit test is ran.
# To speed up the CI time, the resulting environment is cached.
#
# Do not install PyTorch and torchvision here, otherwise they also get cached.

set -e

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
apt-get update && apt-get upgrade -y && apt-get install -y git cmake
# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'
apt-get install -y wget curl \
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
apt-get upgrade -y libstdc++6

cd /usr/lib/x86_64-linux-gnu
ln -s libglut.so.3.12 libglut.so.3
cd $this_dir

root_dir="$(git rev-parse --show-toplevel)"
env_dir="${root_dir}/.venv"

cd "${root_dir}"


# 1. Install uv
printf "* Installing uv
"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"


# 2. Create test environment at ./.venv
printf "python: ${PYTHON_VERSION}
"
printf "* Creating a test environment with uv
"
uv venv "${env_dir}" --python="${PYTHON_VERSION}"
source "${env_dir}/bin/activate"


# 4. Install Conda dependencies
printf "* Installing dependencies (except PyTorch)\n"
echo "  - python=${PYTHON_VERSION}" >> "${this_dir}/environment.yml"
cat "${this_dir}/environment.yml"

uv pip install pip --upgrade

# Dependencies installed via uv pip (see converted script)

# 5. Install StarCraft 2 with SMACv2 maps
starcraft_path="${root_dir}/StarCraftII"
map_dir="${starcraft_path}/Maps"
printf "* Installing StarCraft 2 and SMACv2 maps into ${starcraft_path}\n"
cd "${root_dir}"
# TODO: discuss how we can cache it to avoid downloading ~4 GB on each run.
# e.g adding this into the image learn( which one is used and how it is maintained)
wget https://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
# The archive contains StarCraftII folder. Password comes from the documentation.
unzip -qo -P iagreetotheeula SC2.4.10.zip
mkdir -p "${map_dir}"
# Install Maps
wget https://github.com/oxwhirl/smacv2/releases/download/maps/SMAC_Maps.zip
unzip SMAC_Maps.zip
mkdir "${map_dir}/SMAC_Maps"
mv *.SC2Map "${map_dir}/SMAC_Maps"
printf "StarCraft II and SMAC are installed."
