#!/usr/bin/env bash

# This script is for setting up environment in which unit test is ran.
# To speed up the CI time, the resulting environment is cached.
#
# Do not install PyTorch and torchvision here, otherwise they also get cached.

set -e

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# Avoid error: "fatal: unsafe repository"
apt-get update && apt-get install -y git curl wget gcc g++ cmake
git config --global --add safe.directory '*'
root_dir="$(git rev-parse --show-toplevel)"
env_dir="${root_dir}/.venv"

cd "${root_dir}"


# 1. Install uv
printf "* Installing uv\n"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"


# 2. Create test environment at ./.venv
printf "python: ${PYTHON_VERSION}
"
printf "* Creating a test environment with uv
"
uv venv "${env_dir}" --python="${PYTHON_VERSION}"
source "${env_dir}/bin/activate"


## 3. Install mujoco
#printf "* Installing mujoco and related\n"
#mkdir -p $root_dir/.mujoco
#cd $root_dir/.mujoco/
#wget https://github.com/deepmind/mujoco/releases/download/2.1.1/mujoco-2.1.1-linux-x86_64.tar.gz
#tar -xf mujoco-2.1.1-linux-x86_64.tar.gz
#wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
#tar -xf mujoco210-linux-x86_64.tar.gz
#cd $this_dir

# 4. Install Conda dependencies
printf "* Installing dependencies (except PyTorch)\n"
echo "  - python=${PYTHON_VERSION}" >> "${this_dir}/environment.yml"
cat "${this_dir}/environment.yml"


if [[ $OSTYPE == 'darwin'* ]]; then
  PRIVATE_MUJOCO_GL=glfw
elif [ "${CU_VERSION:-}" == cpu ]; then
  PRIVATE_MUJOCO_GL=osmesa
else
  PRIVATE_MUJOCO_GL=egl
fi

export MUJOCO_GL=$PRIVATE_MUJOCO_GL
export MUJOCO_PY_MUJOCO_PATH=$root_dir/.mujoco/mujoco210
export MAX_IDLE_COUNT=1000
export DISPLAY=:99
export MJLIB_PATH=$root_dir/.mujoco/mujoco-2.1.1/lib/libmujoco.so.2.1.1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$root_dir/.mujoco/mujoco210/bin
export SDL_VIDEODRIVER=dummy
export PYOPENGL_PLATFORM=$PRIVATE_MUJOCO_GL
export TOKENIZERS_PARALLELISM=true

# Software rendering requires GLX and OSMesa.
if [ $PRIVATE_MUJOCO_GL == 'egl' ] || [ $PRIVATE_MUJOCO_GL == 'osmesa' ] ; then
  yum makecache
  yum install -y glfw
  yum install -y glew
  yum install -y mesa-libGL
  yum install -y mesa-libGL-devel
  yum install -y mesa-libOSMesa-devel
  yum -y install egl-utils
  yum -y install freeglut
fi

uv pip install pip --upgrade

# Dependencies installed via uv pip (see converted script)


if [[ $OSTYPE != 'darwin'* ]]; then
  # install ale-py: manylinux names are broken for CentOS so we need to manually download and
  # rename them
  PY_VERSION=$(python --version)
  echo "installing gymnasium"
  uv pip install "gymnasium[atari]>=1.1"
else
  uv pip install "gymnasium[atari]>=1.1"
fi
