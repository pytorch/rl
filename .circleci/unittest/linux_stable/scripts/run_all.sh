#!/usr/bin/env bash

set -euxo pipefail

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
bash ${this_dir}/setup_env.sh
bash ${this_dir}/install.sh
bash ${this_dir}/run_test.sh
bash ${this_dir}/post_process.sh
