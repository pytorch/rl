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
apt-get update && apt-get install -y git wget libglew-dev libx11-dev x11proto-dev g++

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
  conda env remove --prefix ./cloned_env
done

# gym[atari]==0.19 is broken, so we install only gym without dependencies.
for GYM_VERSION in '0.19'
do
  # Create a copy of the conda env and work with this
  conda deactivate
  conda create --prefix ./cloned_env --clone ./env -y
  conda activate ./cloned_env

  echo "Testing gym version: ${GYM_VERSION}"
  pip3 install gym==$GYM_VERSION
  $DIR/run_test.sh

  # delete the conda copy
  conda deactivate
  conda env remove --prefix ./cloned_env
done

# gym[atari]==0.20 installs ale-py==0.8, but this version is not compatible with gym<0.26, so we downgrade it.
for GYM_VERSION in '0.20'
do
  # Create a copy of the conda env and work with this
  conda deactivate
  conda create --prefix ./cloned_env --clone ./env -y
  conda activate ./cloned_env

  echo "Testing gym version: ${GYM_VERSION}"
  pip3 install 'gym[atari]'==$GYM_VERSION
  pip3 install ale-py==0.7
  $DIR/run_test.sh

  # delete the conda copy
  conda deactivate
  conda env remove --prefix ./cloned_env
done

for GYM_VERSION in '0.25'
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
  conda env remove --prefix ./cloned_env
done

# For this version "gym[accept-rom-license]" is required.
for GYM_VERSION in '0.26'
do
  # Create a copy of the conda env and work with this
  conda deactivate
  conda create --prefix ./cloned_env --clone ./env -y
  conda activate ./cloned_env

  echo "Testing gym version: ${GYM_VERSION}"
  pip3 install 'gym[accept-rom-license]'==$GYM_VERSION
  pip3 install 'gym[atari]'==$GYM_VERSION
  pip3 install gym-super-mario-bros
  $DIR/run_test.sh

  # delete the conda copy
  conda deactivate
  conda env remove --prefix ./cloned_env
done

# For this version "gym[accept-rom-license]" is required.
for GYM_VERSION in '0.27'
do
  # Create a copy of the conda env and work with this
  conda deactivate
  conda create --prefix ./cloned_env --clone ./env -y
  conda activate ./cloned_env

  echo "Testing gym version: ${GYM_VERSION}"
  pip3 install 'gymnasium[accept-rom-license]'==$GYM_VERSION


  if [[ $OSTYPE != 'darwin'* ]]; then
    # install ale-py: manylinux names are broken for CentOS so we need to manually download and
    # rename them
    PY_VERSION=$(python --version)
    if [[ $PY_VERSION == *"3.7"* ]]; then
      wget https://files.pythonhosted.org/packages/ab/fd/6615982d9460df7f476cad265af1378057eee9daaa8e0026de4cedbaffbd/ale_py-0.8.0-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
      pip install ale_py-0.8.0-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
      rm ale_py-0.8.0-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    elif [[ $PY_VERSION == *"3.8"* ]]; then
      wget https://files.pythonhosted.org/packages/0f/8a/feed20571a697588bc4bfef05d6a487429c84f31406a52f8af295a0346a2/ale_py-0.8.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
      pip install ale_py-0.8.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
      rm ale_py-0.8.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    elif [[ $PY_VERSION == *"3.9"* ]]; then
      wget https://files.pythonhosted.org/packages/a0/98/4316c1cedd9934f9a91b6e27a9be126043b4445594b40cfa391c8de2e5e8/ale_py-0.8.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
      pip install ale_py-0.8.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
      rm ale_py-0.8.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    elif [[ $PY_VERSION == *"3.10"* ]]; then
      wget https://files.pythonhosted.org/packages/60/1b/3adde7f44f79fcc50d0a00a0643255e48024c4c3977359747d149dc43500/ale_py-0.8.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
      mv ale_py-0.8.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl ale_py-0.8.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
      pip install ale_py-0.8.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
      rm ale_py-0.8.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    fi
    pip install gymnasium[atari]
  else
    pip install gymnasium[atari]
  fi
  pip install mo-gymnasium

  $DIR/run_test.sh

  # delete the conda copy
  conda deactivate
  conda env remove --prefix ./cloned_env
done
