#!/usr/bin/env bash

# Runs a batch of scripts in a row to allow docker run to keep installed libraries
# and env variables across runs.

DIR="$(cd "$(dirname "$0")" && pwd)"

$DIR/install.sh

# gym==0.13 is installed initially.
$DIR/run_test.sh

# 0.19
export GYM_VERSION='0.19'
#$DIR/install_gym.sh # Fix permission denied error.
conda install gym==$GYM_VERSION
pip3 list
$DIR/run_test.sh

# 0.20
#$DIR/install_gym.sh "0.20"
#$DIR/run_test.sh

# 0.25
#$DIR/install_gym.sh "0.25"
#$DIR/run_test.sh

# 0.26
#$DIR/install_gym.sh "0.26"
#$DIR/run_test.sh
