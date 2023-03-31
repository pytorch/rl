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
conda env config vars set MUJOCO_PY_MUJOCO_PATH=$root_dir/.mujoco/mujoco210 \
  DISPLAY=unix:0.0 \
  MJLIB_PATH=$root_dir/.mujoco/mujoco-2.1.1/lib/libmujoco.so.2.1.1 \
  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$root_dir/.mujoco/mujoco210/bin \
  SDL_VIDEODRIVER=dummy \
  MUJOCO_GL=$PRIVATE_MUJOCO_GL \
  PYOPENGL_PLATFORM=$PRIVATE_MUJOCO_GL

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

pip install pip --upgrade

conda env update --file "${this_dir}/environment.yml" --prune

conda deactivate
conda activate "${env_dir}"

if [[ $OSTYPE != 'darwin'* ]]; then
  # install ale-py: manylinux names are broken for CentOS so we need to manually download and
  # rename them
  PY_VERSION=$(python --version)
  echo "installing ale-py for ${PY_PY_VERSION}"
  if [[ $PY_VERSION == *"3.7"* ]]; then
    wget https://files.pythonhosted.org/packages/ab/fd/6615982d9460df7f476cad265af1378057eee9daaa8e0026de4cedbaffbd/ale_py-0.8.0-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    pip install ale_py-0.8.0-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    rm ale_py-0.8.0-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
  elif [[ $PY_VERSION == *"3.8"* ]]; then
    wget https://files.pythonhosted.org/packages/0f/8a/feed20571a697588bc4bfef05d6a487429c84f31406a52f8af295a0346a2/ale_py-0.8.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    pip install ale_py-0.8.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    rm ale_py-0.8.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
  elif [[ $PY_VERSION == *"3.9"* ]]; then
    wget https://files.pythonhosted.org/packages/a0/98/4316c1cedd9934f9a91b6e27a9be126043b4445594b40cfa391c8de2e5e8/ale_py-0.8.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    pip install ale_py-0.8.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    rm ale_py-0.8.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
  elif [[ $PY_VERSION == *"3.10"* ]]; then
    wget https://files.pythonhosted.org/packages/60/1b/3adde7f44f79fcc50d0a00a0643255e48024c4c3977359747d149dc43500/ale_py-0.8.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
    mv ale_py-0.8.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl ale_py-0.8.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    pip install ale_py-0.8.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    rm ale_py-0.8.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
  fi
  echo "installing gymnasium"
  pip install "gymnasium[atari,accept-rom-license]"
  pip install mo-gymnasium[mujoco]  # requires here bc needs mujoco-py
else
  pip install "gymnasium[atari,accept-rom-license]"
  pip install mo-gymnasium[mujoco]  # requires here bc needs mujoco-py
fi
