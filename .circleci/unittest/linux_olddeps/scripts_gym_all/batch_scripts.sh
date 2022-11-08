#!/usr/bin/env bash

# Runs a batch of scripts in a row to allow docker run to keep installed libraries
# and env variables across runs.

DIR="$(cd "$(dirname "$0")" && pwd)"

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

# Install PyTorch and TorchRL.
$DIR/install.sh

set GYM_VERSION='0.13'
# gym==0.13 is installed initially due to environment.yml
echo 'Running tests for gym version: ${GYM_VERSION}'
$DIR/run_test.sh

set GYM_VERSION='0.19'
pip3 install gym==$GYM_VERSION
echo 'Running tests for gym version: ${GYM_VERSION}'
$DIR/run_test.sh

set GYM_VERSION='0.20'
pip3 install gym==$GYM_VERSION
echo 'Running tests for gym version: ${GYM_VERSION}'
$DIR/run_test.sh

set GYM_VERSION='0.25'
pip3 install gym==$GYM_VERSION
echo 'Running tests for gym version: ${GYM_VERSION}'
$DIR/run_test.sh

set GYM_VERSION='0.26'
pip3 install gym==$GYM_VERSION
echo 'Running tests for gym version: ${GYM_VERSION}'
$DIR/run_test.sh
