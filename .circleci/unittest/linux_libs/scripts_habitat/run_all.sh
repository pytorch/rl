#!/usr/bin/env bash

set -euxo pipefail
set -v

# Avoid error: "fatal: unsafe repository"
apt-get update
apt-get install -y git wget
#apt-get install -y gcc c++

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
bash ${this_dir}/setup_env.sh
bash ${this_dir}/install.sh

apt-get install -y freeglut3 freeglut3-dev
bash ${this_dir}/run_test.sh
bash ${this_dir}/post_process.sh
