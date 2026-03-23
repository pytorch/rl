#!/usr/bin/env bash

set -euxo pipefail

# Run all minari test scripts in sequence, sourcing each one to maintain environment state
this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "Running minari tests with uv-based setup..."

# Source each script in sequence to maintain environment state
source "${this_dir}/setup_env.sh"
source "${this_dir}/install.sh"
PYTHON=./env/bin/python bash "$(git rev-parse --show-toplevel)/.github/unittest/helpers/assert_torch_version.sh" "$TORCH_VERSION"
source "${this_dir}/run_test.sh"
source "${this_dir}/post_process.sh"

echo "Minari tests completed successfully!" 
