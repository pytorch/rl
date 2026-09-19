#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
export PIP_NO_INPUT=1 PIP_DISABLE_PIP_VERSION_CHECK=1 UV_NO_PROGRESS=1
python3 -m venv .cache/fql-bootstrap
.cache/fql-bootstrap/bin/python -m pip install --quiet uv==0.12.17
uv=.cache/fql-bootstrap/bin/uv
"$uv" venv --python 3.11.16 --allow-existing .venv
"$uv" pip install --python .venv/bin/python --torch-backend=auto \
    torch==2.14.0 tensordict==0.14.2 \
    gymnasium==1.3.0 mujoco==3.13.0 \
    setuptools setuptools-scm wheel ninja cmake 'pybind11[global]' numpy \
    pre-commit autoflake pytest scipy psutil \
    -r sota-implementations/fql/requirements.txt
"$uv" pip install --python .venv/bin/python --no-sources --no-build-isolation -e .
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA is required"; print(torch.cuda.get_device_properties(0))'
mkdir -p outputs/fql
"$uv" pip freeze --python .venv/bin/python > outputs/fql/environment.txt
