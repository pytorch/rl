#!/usr/bin/env bash

set -euxo pipefail
set -v


apt-get update && apt-get upgrade -y
apt-get install -y vim git wget

apt-get install -y libglfw3 libgl1-mesa-glx libosmesa6 libglew-dev
apt-get install -y libglvnd0 libgl1 libglx0 libegl1 libgles2

apt-get upgrade -y libstdc++6
apt-get dist-upgrade -y
apt-get install -y g++ gcc

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
if [[ $OSTYPE != 'darwin'* ]]; then
  # from cudagl docker image
  cp $this_dir/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
fi

bash ${this_dir}/setup_env.sh
bash ${this_dir}/install.sh

#apt-get install -y freeglut3 freeglut3-dev
bash ${this_dir}/run_test.sh
bash ${this_dir}/post_process.sh
