#!/usr/bin/env bash

set -euxo pipefail
set -v

# ==================================================================================== #
# ================================ Init ============================================== #


export DEBIAN_FRONTEND=noninteractive
export TZ="${TZ:-Etc/UTC}"
ln -snf "/usr/share/zoneinfo/${TZ}" /etc/localtime || true
echo "${TZ}" > /etc/timezone || true

apt-get update
apt-get install -y --no-install-recommends tzdata
dpkg-reconfigure -f noninteractive tzdata || true

apt-get upgrade -y
apt-get install -y vim git wget cmake

apt-get install -y libglfw3 libgl1-mesa-glx libosmesa6 libglew-dev libosmesa6-dev
apt-get install -y libglvnd0 libgl1 libglx0 libegl1 libgles2
apt-get install -y g++ gcc patchelf

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# from cudagl docker image
cp $this_dir/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json


# ==================================================================================== #
# ================================ Setup env ========================================= #

# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'
root_dir="$(git rev-parse --show-toplevel)"
conda_dir="${root_dir}/conda"
env_dir="${root_dir}/env"
lib_dir="${env_dir}/lib"

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

# Verify we have CPython, not PyPy
python_impl=$(python -c "import platform; print(platform.python_implementation())")
if [ "$python_impl" != "CPython" ]; then
    echo "ERROR: Expected CPython but got $python_impl"
    echo "Python executable: $(which python)"
    echo "Python version: $(python --version)"
    exit 1
fi
printf "* Verified Python implementation: %s\n" "$python_impl"

# 3. Install mujoco
printf "* Installing mujoco and related\n"
mkdir -p $root_dir/.mujoco
cd $root_dir/.mujoco/
#wget https://github.com/deepmind/mujoco/releases/download/2.1.1/mujoco-2.1.1-linux-x86_64.tar.gz
#tar -xf mujoco-2.1.1-linux-x86_64.tar.gz
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xf mujoco210-linux-x86_64.tar.gz
cd "${root_dir}"

# 4. Install Conda dependencies
printf "* Installing dependencies (except PyTorch)\n"
# Add python version to environment.yml if not already present (idempotent)
if ! grep -q "python=${PYTHON_VERSION}" "${this_dir}/environment.yml"; then
    echo "  - python=${PYTHON_VERSION}" >> "${this_dir}/environment.yml"
fi
cat "${this_dir}/environment.yml"

export MUJOCO_PY_MUJOCO_PATH=$root_dir/.mujoco/mujoco210
#export MJLIB_PATH=$root_dir/.mujoco/mujoco-2.1.1/lib/libmujoco.so.2.1.1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$root_dir/.mujoco/mujoco210/bin
export SDL_VIDEODRIVER=dummy
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LAZY_LEGACY_OP=False
export COMPOSITE_LP_AGGREGATE=0

conda env config vars set \
  MAX_IDLE_COUNT=1000 \
  MUJOCO_PY_MUJOCO_PATH=$root_dir/.mujoco/mujoco210 \
  DISPLAY=:99 \
  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$root_dir/.mujoco/mujoco210/bin \
  SDL_VIDEODRIVER=dummy \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  BATCHED_PIPE_TIMEOUT=60 \
  TOKENIZERS_PARALLELISM=true

pip install pip --upgrade

conda env update --file "${this_dir}/environment.yml" --prune

conda deactivate
conda activate "${env_dir}"

# install d4rl
pip install free-mujoco-py
pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl

# TODO: move this down -- will break torchrl installation
conda install -y -c conda-forge libstdcxx-ng=12
## find libstdc - search in the env's lib directory first, then fall back to conda packages
STDC_LOC=$(find "${env_dir}/lib" -name "libstdc++.so.6" 2>/dev/null | head -1)
if [ -z "$STDC_LOC" ]; then
    # Fall back to searching in conda packages for libstdcxx-ng specifically
    STDC_LOC=$(find conda/pkgs -path "*libstdcxx*" -name "libstdc++.so.6" 2>/dev/null | head -1)
fi
if [ -z "$STDC_LOC" ]; then
    echo "WARNING: Could not find libstdc++.so.6, skipping LD_PRELOAD"
    conda env config vars set \
      MAX_IDLE_COUNT=1000 \
      TOKENIZERS_PARALLELISM=true
else
    echo "Found libstdc++ at: $STDC_LOC"
    conda env config vars set \
      MAX_IDLE_COUNT=1000 \
      LD_PRELOAD=${STDC_LOC} TOKENIZERS_PARALLELISM=true
fi

# Reactivate environment to apply the new env vars
conda deactivate
conda activate "${env_dir}"

# compile mujoco-py (bc it's done at runtime for whatever reason someone thought it was a good idea)
python -c """import gym;import d4rl"""

# install ale-py: manylinux names are broken for CentOS so we need to manually download and
# rename them

# ============================================================================================ #
# ================================ PyTorch & TorchRL ========================================= #


if [[ ${#CU_VERSION} -eq 4 ]]; then
    CUDA_VERSION="${CU_VERSION:2:1}.${CU_VERSION:3:1}"
elif [[ ${#CU_VERSION} -eq 5 ]]; then
    CUDA_VERSION="${CU_VERSION:2:2}.${CU_VERSION:4:1}"
fi
echo "Using CUDA $CUDA_VERSION as determined by CU_VERSION ($CU_VERSION)"
version="$(python -c "print('.'.join(\"${CUDA_VERSION}\".split('.')[:2]))")"

# submodules
git submodule sync && git submodule update --init --recursive

pip3 install ale-py -U
pip3 install "gym[atari,accept-rom-license]" "gymnasium>=1.1.0" -U

printf "Installing PyTorch with %s\n" "${CU_VERSION}"
if [[ "$TORCH_VERSION" == "nightly" ]]; then
  if [ "${CU_VERSION:-}" == cpu ] ; then
      pip3 install --pre torch torchvision numpy==1.26.4 --index-url https://download.pytorch.org/whl/nightly/cpu -U
  else
      pip3 install --pre torch torchvision numpy==1.26.4 --index-url https://download.pytorch.org/whl/nightly/$CU_VERSION
  fi
elif [[ "$TORCH_VERSION" == "stable" ]]; then
    if [ "${CU_VERSION:-}" == cpu ] ; then
      pip3 install torch torchvision numpy==1.26.4 --index-url https://download.pytorch.org/whl/cpu
  else
      pip3 install torch torchvision numpy==1.26.4 --index-url https://download.pytorch.org/whl/$CU_VERSION
  fi
else
  printf "Failed to install pytorch"
  exit 1
fi

# smoke test
python -c "import functorch"

## install snapshot
#pip install git+https://github.com/pytorch/torchsnapshot

# install tensordict
if [[ "$RELEASE" == 0 ]]; then
  pip3 install git+https://github.com/pytorch/tensordict.git
else
  pip3 install tensordict
fi

printf "* Installing torchrl\n"
python -m pip install -e . --no-build-isolation

# ==================================================================================== #
# ================================ Run tests ========================================= #


bash ${this_dir}/run_test.sh

# ==================================================================================== #
# ================================ Post-proc ========================================= #

bash ${this_dir}/post_process.sh
