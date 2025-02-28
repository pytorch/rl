#!/usr/bin/env bash

set -euxo pipefail

apt update
apt install -y libglfw3 libglfw3-dev libglew-dev libgl1-mesa-glx libgl1-mesa-dev mesa-common-dev libegl1-mesa-dev freeglut3 freeglut3-dev

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
bash ${this_dir}/setup_env.sh
bash ${this_dir}/install.sh
bash ${this_dir}/run_test.sh
bash ${this_dir}/post_process.sh
