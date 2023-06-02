#!/usr/bin/env bash

set -euxo pipefail

bash setup_env.sh
bash install.sh
bash run_test.sh
bash post_process.sh
