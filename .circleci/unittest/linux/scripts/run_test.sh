#!/usr/bin/env bash

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

pip3 install pyopengl --upgrade
unset CUDA_VERSION

unset NCCL_VERSION
unset NVARCH
unset NVIDIA_DRIVER_CAPABILITIES
unset NVIDIA_REQUIRE_CUDA
unset NVIDIA_VISIBLE_DEVICES
unset NV_CUDA_CUDART_DEV_VERSION
unset NV_CUDA_CUDART_VERSION
unset NV_CUDA_LIB_VERSION
unset NV_LIBCUBLAS_DEV_PACKAGE
unset NV_LIBCUBLAS_DEV_PACKAGE_NAME
unset NV_LIBCUBLAS_DEV_VERSION
unset NV_LIBCUBLAS_PACKAGE
unset NV_LIBCUBLAS_PACKAGE_NAME
unset NV_LIBCUBLAS_VERSION
unset NV_LIBNCCL_DEV_PACKAGE
unset NV_LIBNCCL_DEV_PACKAGE_NAME
unset NV_LIBNCCL_DEV_PACKAGE_VERSION
unset NV_LIBNCCL_PACKAGE
unset NV_LIBNCCL_PACKAGE_NAME
unset NV_LIBNCCL_PACKAGE_VERSION
unset NV_LIBNPP_DEV_VERSION
unset NV_LIBNPP_VERSION
unset NV_NVML_DEV_VERSION
unset NV_NVPROF_VERSION
unset NV_NVTX_VERSION

export PYENV_ROOT=/opt/circleci/.pyenv
export PYENV_SHELL=bash
export TERM=xterm-256color

export PYTORCH_TEST_WITH_SLOW='1'
python -m torch.utils.collect_env
# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'

root_dir="$(git rev-parse --show-toplevel)"
env_dir="${root_dir}/env"
lib_dir="${env_dir}/lib"

# solves ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$lib_dir
export MKL_THREADING_LAYER=GNU

echo `printenv`

pytest test/smoke_test.py -v --durations 20
pytest test/smoke_test_deps.py -v --durations 20
pytest --instafail -v --durations 20
