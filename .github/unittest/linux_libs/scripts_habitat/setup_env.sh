#!/usr/bin/env bash

# This script is for setting up environment in which unit test is ran.
# To speed up the CI time, the resulting environment is cached.
#
# Do not install PyTorch and torchvision here, otherwise they also get cached.

set -e
set -v

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# Avoid error: "fatal: unsafe repository"
#apt-get update && apt-get install -y git wget gcc g++ unzip

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

# set debug variables
conda env config vars set \
  MAX_IDLE_COUNT=1000 \
  MAGNUM_LOG=debug HABITAT_SIM_LOG=debug TOKENIZERS_PARALLELISM=true
conda deactivate && conda activate "${env_dir}"

pip3 install "cython<3"
conda install -c anaconda cython="<3.0.0" -y

# 3. Install git LFS (newer version that supports git lfs prune -f)
mkdir -p git_lfs_tmp
cd git_lfs_tmp
wget https://github.com/git-lfs/git-lfs/releases/download/v3.4.0/git-lfs-linux-amd64-v3.4.0.tar.gz
tar -xf git-lfs-linux-amd64-v3.4.0.tar.gz
# The binary is in git-lfs-3.4.0/git-lfs
chmod 755 git-lfs-3.4.0/git-lfs
# Install to /usr/local/bin so it's available system-wide
cp git-lfs-3.4.0/git-lfs /usr/local/bin/
cd "${root_dir}"
git lfs install

# Configure git-lfs for better performance (higher timeouts, more concurrent transfers)
git config --global lfs.activitytimeout 600
git config --global lfs.dialtimeout 60
git config --global lfs.tlstimeout 60
git config --global lfs.concurrenttransfers 8
git config --global http.version HTTP/1.1
rm -rf git_lfs_tmp

# 4. Install Conda dependencies
printf "* Installing dependencies (except PyTorch)\n"
echo "  - python=${PYTHON_VERSION}" >> "${this_dir}/environment.yml"
cat "${this_dir}/environment.yml"

pip install pip --upgrade

conda env update --file "${this_dir}/environment.yml" --prune

# 5. Install habitat-sim from source (conda packages don't support Python 3.10+)
# Install build dependencies
pip3 install ninja numpy

# Clone and build habitat-sim from source
cd "${root_dir}"
git clone --branch stable https://github.com/facebookresearch/habitat-sim.git --recursive
cd habitat-sim

# Build with headless (EGL) and bullet physics support
# Ensure system cmake is used (pip cmake 4.x is incompatible with habitat-sim's CMake files)
# Put /usr/bin at the front of PATH to prefer system cmake over any pip-installed cmake
export PATH="/usr/bin:$PATH"
# Also set CMAKE_EXECUTABLE to explicitly use system cmake
export CMAKE_EXECUTABLE=/usr/bin/cmake
pip3 install . --no-build-isolation

cd "${root_dir}"

# 6. Download required Habitat test datasets manually (faster than datasets_download)
# Using smudge-disabled clone + git lfs pull for better performance
mkdir -p data/versioned_data
mkdir -p data/objects
mkdir -p data/robots

echo "$(date): Starting replica_cad_dataset download..."
git clone --progress --depth 1 --branch v1.6 \
  -c filter.lfs.smudge= -c filter.lfs.required=false \
  https://huggingface.co/datasets/ai-habitat/ReplicaCAD_dataset.git \
  data/versioned_data/replica_cad_dataset
cd data/versioned_data/replica_cad_dataset
time git lfs pull
cd "${root_dir}"

# Create symlink expected by habitat
ln -sf versioned_data/replica_cad_dataset data/replica_cad

echo "$(date): Starting YCB objects download..."
git clone --progress --depth 1 \
  -c filter.lfs.smudge= -c filter.lfs.required=false \
  https://huggingface.co/datasets/ai-habitat/ycb.git \
  data/objects/ycb
cd data/objects/ycb
time git lfs pull
cd "${root_dir}"

echo "$(date): Starting hab_fetch robot download..."
git clone --progress --depth 1 \
  -c filter.lfs.smudge= -c filter.lfs.required=false \
  https://huggingface.co/datasets/ai-habitat/hab_fetch.git \
  data/robots/hab_fetch
cd data/robots/hab_fetch
time git lfs pull
cd "${root_dir}"

echo "$(date): Dataset downloads complete!"
echo "Total data size:"
du -sh data/

# Install habitat-lab
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip3 install -e habitat-lab
pip3 install -e habitat-baselines  # install habitat_baselines
conda run python -m pip install "gym[atari,accept-rom-license]" pygame
