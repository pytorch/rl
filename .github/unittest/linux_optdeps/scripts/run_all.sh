#!/usr/bin/env bash

set -euxo pipefail
set -v

# ==================================================================================== #
# ================================ Init ============================================== #


if [[ $OSTYPE != 'darwin'* ]]; then
  apt-get update && apt-get upgrade -y
  apt-get install -y vim git wget

  apt-get install -y libglfw3 libgl1-mesa-glx libosmesa6 libglew-dev
  apt-get install -y libglvnd0 libgl1 libglx0 libegl1 libgles2

  if [ "${CU_VERSION:-}" == cpu ] ; then
    # solves version `GLIBCXX_3.4.29' not found for tensorboard
#    apt-get install -y gcc-4.9
    apt-get upgrade -y libstdc++6
    apt-get dist-upgrade -y
  else
    apt-get install -y g++ gcc
  fi

fi

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
if [[ $OSTYPE != 'darwin'* ]]; then
  # from cudagl docker image
  cp $this_dir/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
fi


# ==================================================================================== #
# ================================ Setup env ========================================= #

bash ${this_dir}/setup_env.sh

# ============================================================================================ #
# ================================ PyTorch & TorchRL ========================================= #

bash ${this_dir}/install.sh

# ==================================================================================== #
# ================================ Run tests ========================================= #


bash ${this_dir}/run_test.sh

# ==================================================================================== #
# ================================ Post-proc ========================================= #

bash ${this_dir}/post_process.sh
