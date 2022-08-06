#!/usr/bin/env bash

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

pip3 install pyopengl --upgrade
#unset CUDA_VERSION
#
#unset NCCL_VERSION
#unset NVARCH
#unset NVIDIA_DRIVER_CAPABILITIES
#unset NVIDIA_REQUIRE_CUDA
#unset NVIDIA_VISIBLE_DEVICES
#unset NV_CUDA_CUDART_DEV_VERSION
#unset NV_CUDA_CUDART_VERSION
#unset NV_CUDA_LIB_VERSION
#unset NV_LIBCUBLAS_DEV_PACKAGE
#unset NV_LIBCUBLAS_DEV_PACKAGE_NAME
#unset NV_LIBCUBLAS_DEV_VERSION
#unset NV_LIBCUBLAS_PACKAGE
#unset NV_LIBCUBLAS_PACKAGE_NAME
#unset NV_LIBCUBLAS_VERSION
#unset NV_LIBNCCL_DEV_PACKAGE
#unset NV_LIBNCCL_DEV_PACKAGE_NAME
#unset NV_LIBNCCL_DEV_PACKAGE_VERSION
#unset NV_LIBNCCL_PACKAGE
#unset NV_LIBNCCL_PACKAGE_NAME
#unset NV_LIBNCCL_PACKAGE_VERSION
#unset NV_LIBNPP_DEV_VERSION
#unset NV_LIBNPP_VERSION
#unset NV_NVML_DEV_VERSION
#unset NV_NVPROF_VERSION
#unset NV_NVTX_VERSION
#
#export PYENV_ROOT=/opt/circleci/.pyenv
#export PYENV_SHELL=bash
#export TERM=xterm-256color
#export DBUS_SESSION_BUS_ADDRESS=/dev/null
#export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
#export JDK_HOME=/usr/lib/jvm/java-11-openjdk-amd64
#export JRE_HOME=/usr/lib/jvm/java-11-openjdk-amd64
#export M2_HOME=/usr/local/apache-maven
#export MAVEN_OPTS=-Xmx2048m
#export PATH=/home/circleci/project/env/bin:/home/circleci/project/conda/condabin:/home/circleci/.yarn/bin:/home/circleci/.config/yarn/global/node_modules/.bin:/opt/android/sdk/ndk/23.0.7599858:/opt/android/sdk/emulator:/opt/android/sdk/cmdline-tools/latest/bin:/opt/android/sdk/tools:/opt/android/sdk/tools/bin:/opt/android/sdk/platform-tools:/opt/android/sdk/platform-tools/bin:/home/circleci/.go_workspace/bin:/usr/local/go/bin:/opt/circleci/.pyenv/shims:/opt/circleci/.pyenv/bin:/opt/google/google-cloud-sdk/bin:/usr/local/apache-maven/bin:/home/circleci/bin:/home/circleci/.yarn/bin:/home/circleci/.config/yarn/global/node_modules/.bin:/opt/android/sdk/ndk/23.0.7599858:/opt/android/sdk/emulator:/opt/android/sdk/cmdline-tools/latest/bin:/opt/android/sdk/tools:/opt/android/sdk/tools/bin:/opt/android/sdk/platform-tools:/opt/android/sdk/platform-tools/bin:/opt/circleci/.rvm/gems/ruby-3.0.2/bin:/opt/circleci/.rvm/gems/ruby-3.0.2@global/bin:/opt/circleci/.rvm/rubies/ruby-3.0.2/bin:/home/circleci/.go_workspace/bin:/usr/local/go/bin:/opt/circleci/.nvm/versions/node/v14.17.3/bin:/opt/circleci/.pyenv/shims:/opt/circleci/.pyenv/bin:/opt/google/google-cloud-sdk/bin:/usr/local/apache-maven/bin:/home/circleci/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin:/usr/local/gradle-7.1.1/bin:/opt/circleci/.rvm/bin:/snap/bin:/opt/circleci/.rvm/bin:/usr/local/gradle-7.1.1/bin:/snap/bin:/opt/circleci/.rvm/bin
#
#unset LIBRARY_PATH
#export __CONDA_SHLVL_1_DISPLAY=:99
#unset __CONDA_SHLVL_1_LD_LIBRARY_PATH

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
