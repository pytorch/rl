#!/usr/bin/env bash

set -euxo pipefail
set -v


apt-get update && apt-get upgrade -y
apt-get install -y vim git wget cmake unzip

apt-get install -y libglfw3 libgl1-mesa-glx libosmesa6 libglew-dev
apt-get install -y libglvnd0 libgl1 libglx0 libegl1 libgles2

apt-get install -y g++ gcc
#apt-get upgrade -y libstdc++6
#apt-get install -y libgcc
apt-get dist-upgrade -y

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# from cudagl docker image
cp $this_dir/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

bash ${this_dir}/setup_env.sh
bash ${this_dir}/install.sh

# Download required Habitat datasets
echo "Starting Habitat dataset download..."
if bash ${this_dir}/download_datasets.sh; then
    echo "Habitat dataset download completed successfully!"
else
    echo "ERROR: Habitat dataset download failed!"
    echo "Checking what was downloaded:"
    ls -la data/ 2>/dev/null || echo "No data directory found"
    ls -la data/scene_datasets/ 2>/dev/null || echo "No scene_datasets directory found"
    ls -la data/datasets/ 2>/dev/null || echo "No datasets directory found"
    exit 1
fi

#apt-get install -y freeglut3 freeglut3-dev
bash ${this_dir}/run_test.sh
bash ${this_dir}/post_process.sh
