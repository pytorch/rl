#!/usr/bin/env bash

unset PYTORCH_VERSION
# For unittest, nightly PyTorch is used as the following section,
# so no need to set PYTORCH_VERSION.
# In fact, keeping PYTORCH_VERSION forces us to hardcode PyTorch version in config.

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env


echo "before"
echo $LD_LIBRARY_PATH

mkdir -p ./temp-packages
yum install -y dnf
# sudo dnf --installroot=./temp-packages makecache
dnf --installroot=./temp-packages install -y glfw
dnf --installroot=./temp-packages install -y glew
dnf --installroot=./temp-packages install -y mesa-libGL
dnf --installroot=./temp-packages install -y mesa-libGL-devel
dnf --installroot=./temp-packages install -y mesa-libOSMesa-devel
dnf --installroot=./temp-packages -y install egl-utils
dnf --installroot=./temp-packages -y install freeglut

# conda env config vars set LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./env/x86_64-conda-linux-gnu/sysroot/usr/lib64/

conda deactivate
conda activate ./env


echo "after"
echo $LD_LIBRARY_PATH
echo "$(ldconfig -p | grep libEGL.so.1 | tr ' ' '\n' | grep /)"
echo "$(ldconfig -p | grep libGL.so.1 | tr ' ' '\n' | grep /)"
echo "$(ldconfig -p | grep libOpenGL.so.1 | tr ' ' '\n' | grep /)"
# suggested by habitat team: do we need this?
#conda create -n $MY_TEST_ENV python=3.7 cmake=3.14.0 -y
#conda activate $MY_TEST_ENV

# If you already have an environment you want to use, you can just run the following:
conda install habitat-sim withbullet headless -c conda-forge -c aihabitat-nightly -y
conda run python -m pip install install git+https://github.com/facebookresearch/habitat-lab.git#subdirectory=habitat-lab
conda run python -m pip install install "gym[atari,accept-rom-license]" pygame

# smoke test
python -c "import habitat;import habitat.utils.gym_definitions"
