#!/usr/bin/env bash

set -euxo pipefail

apt update
# libgl1-mesa-glx and freeglut3 are gone on noble; libgl1 + libglx-mesa0 is
# what the former became, and freeglut3-dev pulls the runtime. All of these
# resolve on jammy too, so the list works on either image.
apt install -y libglfw3 libglfw3-dev libglew-dev libgl1 libglx-mesa0 libgl1-mesa-dev mesa-common-dev libegl1-mesa-dev freeglut3-dev

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
bash ${this_dir}/setup_env.sh
bash ${this_dir}/install.sh
PYTHON=./env/bin/python bash "$(git rev-parse --show-toplevel)/.github/unittest/helpers/assert_torch_version.sh" "$TORCH_VERSION"
bash ${this_dir}/run_test.sh
bash ${this_dir}/post_process.sh
