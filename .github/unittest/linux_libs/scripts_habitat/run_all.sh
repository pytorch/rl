#!/usr/bin/env bash

set -euxo pipefail
set -v


apt-get update && apt-get upgrade -y
apt-get install -y vim git wget cmake ninja-build

# OpenGL/EGL dependencies for headless rendering
apt-get install -y libglfw3 libglfw3-dev libgl1-mesa-glx libosmesa6 libosmesa6-dev libglew-dev
apt-get install -y libglvnd0 libgl1 libglx0 libegl1 libgles2
apt-get install -y libegl1-mesa-dev libgles2-mesa-dev

# Build tools and libraries for habitat-sim
apt-get install -y g++ gcc
apt-get install -y libjpeg-dev libpng-dev
apt-get install -y pkg-config

#apt-get upgrade -y libstdc++6
#apt-get install -y libgcc
apt-get dist-upgrade -y

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# from cudagl docker image
cp $this_dir/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

bash ${this_dir}/setup_env.sh
bash ${this_dir}/install.sh

#apt-get install -y freeglut3 freeglut3-dev
bash ${this_dir}/run_test.sh
bash ${this_dir}/post_process.sh
