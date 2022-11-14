#!/usr/bin/env bash

# Runs a batch of scripts in a row to allow docker run to keep installed libraries
# and env variables across runs.

DIR="$(cd "$(dirname "$0")" && pwd)"

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

# Install PyTorch and TorchRL.
$DIR/install.sh

# Extracted from run_test.sh to run once.
yum makecache && yum install libglvnd-devel mesa-libGL mesa-libGL-devel mesa-libEGL mesa-libEGL-devel glfw mesa-libOSMesa-devel glew glew-devel egl-utils freeglut xorg-x11-server-Xvfb -y

# This version is installed initially (see environment.yml)
for GYM_VERSION in '0.13'
do
  echo "Installing gym version: ${GYM_VERSION}"
  pip3 install 'gym[atari]'==$GYM_VERSION
  echo "Running tests for gym version: ${GYM_VERSION}"
  $DIR/run_test.sh
done

# gym[atari]==0.19 is broken, so we install only gym without dependencies.
for GYM_VERSION in '0.19'
do
  echo "Installing gym version: ${GYM_VERSION}"
  pip3 install gym==$GYM_VERSION
  echo "Running tests for gym version: ${GYM_VERSION}"
  $DIR/run_test.sh
done

# gym[atari]==0.20 installs ale-py==0.8, but this version is not compatible with gym<0.26, so we downgrade it.
for GYM_VERSION in '0.20'
do
  echo "Installing gym version: ${GYM_VERSION}"
  pip3 install 'gym[atari]'==$GYM_VERSION
  pip3 install ale-py==0.7
  echo "Running tests for gym version: ${GYM_VERSION}"
  $DIR/run_test.sh
done

for GYM_VERSION in '0.25'
do
  echo "Installing gym version: ${GYM_VERSION}"
  pip3 install 'gym[atari]'==$GYM_VERSION
  echo "Running tests for gym version: ${GYM_VERSION}"
  $DIR/run_test.sh
done

for GYM_VERSION in '0.26'
do
  echo "Installing gym version: ${GYM_VERSION}"
  pip3 install 'gym[accept-rom-license]'==$GYM_VERSION
  pip3 install 'gym[atari]'==$GYM_VERSION
  echo "Running tests for gym version: ${GYM_VERSION}"
  $DIR/run_test.sh
done