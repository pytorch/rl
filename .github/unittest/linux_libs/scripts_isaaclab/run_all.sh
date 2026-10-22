#!/usr/bin/env bash

set -euxo pipefail

export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES
export PRIVACY_CONSENT=Y
export BATCHED_PIPE_TIMEOUT=60
export TD_GET_DEFAULTS_TO_NONE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONNOUSERSITE=1
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

nvidia-smi

apt-get update
apt-get install -y --no-install-recommends git gcc g++ python3.12 python3.12-dev python3.12-venv \
  libgl1 libegl1 libglu1-mesa

git config --global --add safe.directory '*'
root_dir="$(git rev-parse --show-toplevel)"
cd "${root_dir}"

python3.12 -m venv "${root_dir}/env"
source "${root_dir}/env/bin/activate"
python -m pip install --upgrade pip

printf "* Installing Isaac Lab\n"
python -m pip install "isaaclab==${ISAACLAB_VERSION:-3.0.0}"
python - <<'PY'
import importlib.metadata as metadata

import torch

isaaclab_version = metadata.version("isaaclab")
print(f"isaaclab {isaaclab_version}, torch {torch.__version__} (CUDA {torch.version.cuda})")
assert isaaclab_version.startswith("3."), isaaclab_version
assert torch.version.cuda and torch.version.cuda.startswith("13."), torch.version.cuda
assert torch.cuda.is_available()
PY

printf "* Installing tensordict\n"
if [[ "${RELEASE:-0}" == 0 ]]; then
  python -m pip install --no-deps git+https://github.com/pytorch/tensordict.git
else
  python -m pip install --no-deps "tensordict>=0.15.2,<0.16.0"
fi
python -m pip install cloudpickle packaging importlib_metadata "pyvers>=0.2.3,<0.3.0" "hoptorch>=0.1.4" "pybind11[global]" ninja

printf "* Installing torchrl\n"
python -m pip install -e . --no-build-isolation --no-deps
python -c "import torchrl"

python -m pip install pytest pytest-timeout pytest-rerunfailures ray

TEST_IDS=$(python -m pytest test/libs/test_isaac.py -k isaaclab --collect-only -q 2>/dev/null | grep "::")
if [[ -z "${TEST_IDS}" ]]; then
  echo "ERROR: no Isaac Lab tests collected"
  exit 1
fi

OVERALL=0
while IFS= read -r TEST_ID; do
  [[ -z "${TEST_ID}" ]] && continue
  echo "Running: ${TEST_ID}"
  if ! python -m pytest "${TEST_ID}" -s --timeout=900; then
    echo "TEST FAILED: ${TEST_ID}"
    OVERALL=1
  fi
done <<< "${TEST_IDS}"
exit "${OVERALL}"
