#!/usr/bin/env bash

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env


export PYTORCH_TEST_WITH_SLOW='1'
python -m torch.utils.collect_env
# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'

root_dir="$(git rev-parse --show-toplevel)"
env_dir="${root_dir}/env"
lib_dir="${env_dir}/lib"

## install habitat deps
#yum install -y glfw
#yum install -y glew
#yum install -y mesa-libGL
#yum install -y mesa-libGL-devel
#yum install -y mesa-libOSMesa-devel
#yum -y install egl-utils
#yum -y install freeglut

# If you already have an environment you want to use, you can just run the following:
conda install habitat-sim withbullet headless -c conda-forge -c aihabitat-nightly -y
conda run python -m pip install install git+https://github.com/facebookresearch/habitat-lab.git#subdirectory=habitat-lab
conda run python -m pip install install "gym[atari,accept-rom-license]" pygame

# smoke test
python -c "import habitat;import habitat.utils.gym_definitions"

# solves ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$lib_dir
export MKL_THREADING_LAYER=GNU

# this workflow only tests the libs
python -c "import habitat;import habitat.utils.gym_definitions"
#pytest test/test_libs.py --instafail -v --durations 20
python test/test_libs.py -k test_habitat
