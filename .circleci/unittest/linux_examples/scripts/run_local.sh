#!/bin/bash

set -e

# Read script from line 29
filename=".circleci/unittest/linux_examples/scripts/run_test.sh"
start_line=29
script=$(tail -n +$start_line "$filename")
script="set -e"$'\n'"$script"

# Replace "cuda:0" with "cpu"
script="${script//cuda:0/cpu}"

# Remove any instances of ".circleci/unittest/helpers/coverage_run_parallel.py"
script="${script//.circleci\/unittest\/helpers\/coverage_run_parallel.py}"
script="${script//coverage combine}"
script="${script//coverage xml -i}"

# Execute the modified script
echo "$script" | bash
