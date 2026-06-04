#!/usr/bin/env bash

# Runs a batch of scripts in a row to allow docker run to keep installed libraries
# and env variables across runs.

DIR="$(cd "$(dirname "$0")" && pwd)"

$DIR/install.sh
PYTHON=./env/bin/python bash "$(git rev-parse --show-toplevel)/.github/unittest/helpers/assert_torch_version.sh" stable
if [[ "$RELEASE" == 0 ]]; then
  tensordict_expectation=main
else
  tensordict_expectation=stable
fi
PYTHON=./env/bin/python bash "$(git rev-parse --show-toplevel)/.github/unittest/helpers/assert_torch_tensordict_versions.sh" stable "$tensordict_expectation"
$DIR/run_test.sh
