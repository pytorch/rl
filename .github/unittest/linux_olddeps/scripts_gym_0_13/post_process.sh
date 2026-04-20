#!/usr/bin/env bash

set -e
set -v

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env
