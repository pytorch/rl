#!/usr/bin/env bash

# Runs a batch of scripts in a row to allow docker run to keep installed libraries
# and env variables across runs.

DIR="$(cd "$(dirname "$0")" && pwd)"

set -e
set -v

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

# Install PyTorch and TorchRL.
$DIR/install.sh

# Extracted from run_test.sh to run once.
apt-get update && apt-get install -y git wget libglew-dev libx11-dev x11proto-dev g++

# solves "'extras_require' must be a dictionary"
pip install setuptools==65.3.0

#mkdir -p third_party
#cd third_party
#git clone https://github.com/vmoens/gym
#cd ..

# This version is installed initially (see environment.yml)
for GYM_VERSION in '0.13'
do
  # Create a copy of the conda env and work with this
  conda deactivate
  conda create --prefix ./cloned_env --clone ./env -y
  conda activate ./cloned_env

  echo "Testing gym version: ${GYM_VERSION}"
  pip3 install 'gym[atari]'==$GYM_VERSION
  $DIR/run_test.sh

  # delete the conda copy
  conda deactivate
  conda env remove --prefix ./cloned_env -y
done

# gym[atari]==0.19 is broken, so we install only gym without dependencies.
for GYM_VERSION in '0.19'
do
  # Create a copy of the conda env and work with this
  conda deactivate
  conda create --prefix ./cloned_env --clone ./env -y
  conda activate ./cloned_env

  echo "Testing gym version: ${GYM_VERSION}"
  # handling https://github.com/openai/gym/issues/3202
  pip3 install wheel==0.38.4
  pip3 install "pip<24.1"
  pip3 install gym==$GYM_VERSION
  $DIR/run_test.sh

  # delete the conda copy
  conda deactivate
  conda env remove --prefix ./cloned_env -y
done

# gym[atari]==0.20 installs ale-py==0.8, but this version is not compatible with gym<0.26, so we downgrade it.
for GYM_VERSION in '0.20'
do
  # Create a copy of the conda env and work with this
  conda deactivate
  conda create --prefix ./cloned_env --clone ./env -y
  conda activate ./cloned_env

  echo "Testing gym version: ${GYM_VERSION}"
  pip3 install wheel==0.38.4
  pip3 install "pip<24.1"
  pip3 install 'gym[atari]'==$GYM_VERSION
  pip3 install ale-py==0.7
  $DIR/run_test.sh

  # delete the conda copy
  conda deactivate
  conda env remove --prefix ./cloned_env -y
done

for GYM_VERSION in '0.25'
do
  # Create a copy of the conda env and work with this
  conda deactivate
  conda create --prefix ./cloned_env --clone ./env -y
  conda activate ./cloned_env

  echo "Testing gym version: ${GYM_VERSION}"
  pip3 install 'gym[atari]'==$GYM_VERSION
  pip3 install pip -U
  $DIR/run_test.sh

  # delete the conda copy
  conda deactivate
  conda env remove --prefix ./cloned_env -y
done

# For this version "gym[accept-rom-license]" is required.
for GYM_VERSION in '0.26'
do
  # Create a copy of the conda env and work with this
  conda deactivate
  conda create --prefix ./cloned_env --clone ./env -y
  conda activate ./cloned_env

  echo "Testing gym version: ${GYM_VERSION}"
  pip3 install 'gym[atari,accept-rom-license]'==$GYM_VERSION
  pip3 install gym-super-mario-bros
  $DIR/run_test.sh

  # delete the conda copy
  conda deactivate
  conda env remove --prefix ./cloned_env -y
done

# For this version "gym[accept-rom-license]" is required.
for GYM_VERSION in '0.27' '0.28'
do
  # Create a copy of the conda env and work with this
  conda deactivate
  conda create --prefix ./cloned_env --clone ./env -y
  conda activate ./cloned_env

  echo "Testing gym version: ${GYM_VERSION}"
  pip3 install 'gymnasium[atari,ale-py]'==$GYM_VERSION

  $DIR/run_test.sh

  # delete the conda copy
  conda deactivate
  conda env remove --prefix ./cloned_env -y
done

# Prev gymnasium
conda deactivate
conda create --prefix ./cloned_env --clone ./env -y
conda activate ./cloned_env

pip3 install 'gymnasium[ale-py,atari]>=1.1.0' mo-gymnasium gymnasium-robotics -U

$DIR/run_test.sh

# delete the conda copy
conda deactivate
conda env remove --prefix ./cloned_env -y

# Skip 1.0.0

# Latest gymnasium
conda deactivate
conda create --prefix ./cloned_env --clone ./env -y
conda activate ./cloned_env

pip3 install 'gymnasium[ale-py,atari]>=1.1.0' mo-gymnasium gymnasium-robotics -U

$DIR/run_test.sh

# delete the conda copy
conda deactivate
conda env remove --prefix ./cloned_env -y
