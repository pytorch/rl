#!/usr/bin/env bash

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

unset NV_LIBNCCL_PACKAGE_NAME
unset NV_LIBCUBLAS_DEV_VERSION
unset NV_LIBNPP_VERSION
unset NV_LIBNPP_DEV_VERSION
unset NV_CUDA_LIB_VERSION
unset NV_NVML_DEV_VERSION
unset NVIDIA_VISIBLE_DEVICES
unset NV_LIBNCCL_PACKAGE_VERSION
unset NV_NVTX_VERSION
unset NV_NVPROF_VERSION
unset NVIDIA_DRIVER_CAPABILITIES
unset NV_LIBNCCL_DEV_PACKAGE
unset NV_LIBCUBLAS_VERSION
unset NV_LIBNCCL_DEV_PACKAGE_VERSION
unset NV_CUDA_CUDART_VERSION
unset __CONDA_SHLVL_1_LD_LIBRARY_PATH
unset NVARCH
unset NV_CUDA_CUDART_DEV_VERSION
unset CUDA_VERSION
unset NV_LIBCUBLAS_PACKAGE
unset NV_LIBCUBLAS_PACKAGE_NAME
unset NCCL_VERSION
unset NV_LIBNCCL_PACKAGE
unset NV_LIBCUBLAS_DEV_PACKAGE_NAME
unset NV_LIBNCCL_DEV_PACKAGE_NAME
unset NV_LIBCUBLAS_DEV_PACKAGE

pip3 install pyrender
pip3 install pyopengl --upgrade
pip3 uninstall dm_control
pip3 install dm_control

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export __GL_SHADER_DISK_CACHE=0
export __GL_SHADER_DISK_CACHE_PATH=/tmp
printf "DISPLAY:$DISPLAY-->\n"

export PYTORCH_TEST_WITH_SLOW='1'
python -m torch.utils.collect_env
# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'

root_dir="$(git rev-parse --show-toplevel)"
env_dir="${root_dir}/env"
lib_dir="${env_dir}/lib"

# solves ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$lib_dir
printf "LD_LIBRARY_PATH:$LD_LIBRARY_PATH-->\n"
export MKL_THREADING_LAYER=GNU

printenv

pytest test/smoke_test.py -v --durations 20
pytest test/smoke_test_deps.py -v --durations 20
pytest --instafail -v --durations 20
