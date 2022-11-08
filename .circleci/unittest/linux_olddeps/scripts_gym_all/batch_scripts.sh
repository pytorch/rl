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

# gym[atari]<0.25 installs ale-py==0.8 by default, but this version is not compatible with gym<0.26
pip3 install ale-py==0.7

for GYM_VERSION in '0.13' '0.19' '0.20' '0.25' '0.26'
do
  echo "Installing gym version: ${GYM_VERSION}"
  pip3 install 'gym[atari]'==$GYM_VERSION
  echo "Running tests for gym version: ${GYM_VERSION}"
  $DIR/run_test.sh
done
