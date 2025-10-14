#!/usr/bin/env bash

# Runs a batch of scripts in a row to allow docker run to keep installed libraries
# and env variables across runs.

DIR="$(cd "$(dirname "$0")" && pwd)"

set -e
set -v

root_dir="$(git rev-parse --show-toplevel)"
source "${root_dir}/.venv/bin/activate"

# Install PyTorch and TorchRL.
$DIR/install.sh

# Extracted from run_test.sh to run once.
apt-get update && apt-get install -y git wget libglew-dev libx11-dev x11proto-dev g++

# solves "'extras_require' must be a dictionary"
uv pip install setuptools==65.3.0

# Helper function to test with a specific gym version
test_gym_version() {
  local GYM_VERSION=$1
  local EXTRA_INSTALL=$2
  
  echo "Testing gym version: ${GYM_VERSION}"
  
  # Install the specific gym version in the current environment
  eval "$EXTRA_INSTALL"
  
  $DIR/run_test.sh
  
  # Uninstall gym to avoid conflicts with next version
  uv pip uninstall gym gymnasium -y 2>/dev/null || true
}

# This version is installed initially
for GYM_VERSION in '0.13'
do
  test_gym_version "$GYM_VERSION" "uv pip install 'gym[atari]==$GYM_VERSION'"
done

# gym[atari]==0.19 is broken, so we install only gym without dependencies.
for GYM_VERSION in '0.19'
do
  test_gym_version "$GYM_VERSION" "uv pip install wheel==0.38.4 && uv pip install 'pip<24.1' && uv pip install gym==$GYM_VERSION"
done

# gym[atari]==0.20 installs ale-py==0.8, but this version is not compatible with gym<0.26, so we downgrade it.
for GYM_VERSION in '0.20'
do
  test_gym_version "$GYM_VERSION" "uv pip install wheel==0.38.4 && uv pip install 'pip<24.1' && uv pip install 'gym[atari]==$GYM_VERSION' && uv pip install ale-py==0.7"
done

for GYM_VERSION in '0.25'
do
  test_gym_version "$GYM_VERSION" "uv pip install 'gym[atari]==$GYM_VERSION'"
done

# For this version "gym[accept-rom-license]" is required.
for GYM_VERSION in '0.26'
do
  test_gym_version "$GYM_VERSION" "uv pip install 'gym[atari,accept-rom-license]==$GYM_VERSION' && uv pip install gym-super-mario-bros"
done

# For this version "gym[accept-rom-license]" is required.
for GYM_VERSION in '0.27' '0.28'
do
  test_gym_version "$GYM_VERSION" "uv pip install 'gymnasium[atari,ale-py]==$GYM_VERSION'"
done

# Prev gymnasium
echo "Testing gymnasium >= 1.1.0"
uv pip install 'gymnasium[ale-py,atari]>=1.1.0' mo-gymnasium gymnasium-robotics -U
$DIR/run_test.sh
uv pip uninstall gymnasium -y 2>/dev/null || true

# Latest gymnasium
echo "Testing latest gymnasium"
uv pip install 'gymnasium[ale-py,atari]>=1.1.0' mo-gymnasium gymnasium-robotics -U
$DIR/run_test.sh
