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

apt-get install -y libglfw3 libosmesa6 libglew-dev libosmesa6-dev
apt-get install -y libglvnd0 libgl1 libglx0 libglx-mesa0 libegl1 libgles2
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
    # Force CPython from the main conda channels (avoid GraalPy).
    conda create --override-channels -c defaults -c pytorch --prefix "${env_dir}" -y python="$PYTHON_VERSION"
fi
conda activate "${env_dir}"

# Verify we're running CPython (wheels won't work on GraalPy)
python -c "import sys; assert sys.implementation.name == 'cpython', f'Expected CPython, got {sys.implementation.name}'"

# 3. Install mujoco

# 4. Install Conda dependencies
printf "* Installing dependencies (except PyTorch)\n"
# Add python version to environment.yml if not already present (idempotent)
if ! grep -q "python=${PYTHON_VERSION}" "${this_dir}/environment.yml"; then
    echo "  - python=${PYTHON_VERSION}" >> "${this_dir}/environment.yml"
fi
cat "${this_dir}/environment.yml"

export SDL_VIDEODRIVER=dummy
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LAZY_LEGACY_OP=False
export COMPOSITE_LP_AGGREGATE=0

conda env config vars set \
  MAX_IDLE_COUNT=1000 \
  DISPLAY=:99 \
  SDL_VIDEODRIVER=dummy \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  BATCHED_PIPE_TIMEOUT=60 \
  TOKENIZERS_PARALLELISM=true

# Use python -m pip to ensure we use conda's Python, not system GraalPy
python -m pip install pip --upgrade

conda env update --file "${this_dir}/environment.yml" --prune

conda deactivate
conda activate "${env_dir}"

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

# ============================================================================================ #
# ================================ PyTorch & TorchRL ========================================= #


if [[ ${#CU_VERSION} -eq 4 ]]; then
    CUDA_VERSION="${CU_VERSION:2:1}.${CU_VERSION:3:1}"
elif [[ ${#CU_VERSION} -eq 5 ]]; then
    CUDA_VERSION="${CU_VERSION:2:2}.${CU_VERSION:4:1}"
fi
echo "Using CUDA $CUDA_VERSION as determined by CU_VERSION ($CU_VERSION)"
# submodules
git submodule sync && git submodule update --init --recursive

# Gymnasium Atari support pulls ale-py (+ ROMs) as needed.
python -m pip install -U "gymnasium[atari,accept-rom-license,mujoco]>=1.1.0"

printf "Installing PyTorch with %s\n" "${CU_VERSION}"
if [[ "$TORCH_VERSION" == "nightly" ]]; then
  if [ "${CU_VERSION:-}" == cpu ] ; then
      python -m pip install --pre torch torchvision numpy==1.26.4 --index-url https://download.pytorch.org/whl/nightly/cpu -U
  else
      python -m pip install --pre torch torchvision numpy==1.26.4 --index-url https://download.pytorch.org/whl/nightly/$CU_VERSION
  fi
elif [[ "$TORCH_VERSION" == "stable" ]]; then
    if [ "${CU_VERSION:-}" == cpu ] ; then
      python -m pip install torch torchvision numpy==1.26.4 --index-url https://download.pytorch.org/whl/cpu
  else
      python -m pip install torch torchvision numpy==1.26.4 --index-url https://download.pytorch.org/whl/$CU_VERSION
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
  python -m pip install git+https://github.com/pytorch/tensordict.git
else
  python -m pip install tensordict
fi

printf "* Installing torchrl\n"
python -m pip install -e . --no-build-isolation

# ==================================================================================== #
# ================================ Run tests ========================================= #


bash ${this_dir}/run_test.sh

# ==================================================================================== #
# ================================ Post-proc ========================================= #

bash ${this_dir}/post_process.sh
