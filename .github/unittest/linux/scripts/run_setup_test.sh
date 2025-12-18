#!/usr/bin/env bash

set -euxo pipefail

if [[ $OSTYPE != 'darwin'* ]]; then
  export DEBIAN_FRONTEND=noninteractive
  export TZ="${TZ:-Etc/UTC}"
  ln -snf "/usr/share/zoneinfo/${TZ}" /etc/localtime || true
  echo "${TZ}" > /etc/timezone || true

  apt-get update
  apt-get install -y --no-install-recommends tzdata
  dpkg-reconfigure -f noninteractive tzdata || true

  apt-get upgrade -y
  apt-get install -y git wget cmake curl python3-dev g++ gcc
fi

# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'
root_dir="$(git rev-parse --show-toplevel)"
env_dir="${root_dir}/venv-setup-test"

cd "${root_dir}"

# Install uv (used for --no-deps install path parity with CI)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

rm -rf "${env_dir}"
uv venv --python "${PYTHON_VERSION}" "${env_dir}"
source "${env_dir}/bin/activate"

uv_pip_install() {
  uv pip install --no-progress --python "${env_dir}/bin/python" "$@"
}

python -c "import sys; print(sys.version)"

# Ensure `python -m pip` exists (uv-created venvs may not include pip).
python -m ensurepip --upgrade

# Minimal runtime/build deps + pytest only.
uv_pip_install \
  pytest \
  setuptools \
  wheel \
  packaging \
  cloudpickle \
  pyvers \
  numpy \
  ninja \
  "pybind11[global]>=2.13"

ref_name="${GITHUB_REF_NAME:-}"
if [[ -z "${ref_name}" && -n "${GITHUB_REF:-}" ]]; then
  ref_name="${GITHUB_REF#refs/heads/}"
fi

if [[ "${ref_name}" == release/* ]]; then
  export RELEASE=1
  export TORCH_VERSION=stable
else
  export RELEASE=0
  export TORCH_VERSION=nightly
fi

if [[ "$TORCH_VERSION" == "nightly" ]]; then
  uv_pip_install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
else
  uv_pip_install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# tensordict is a hard dependency of torchrl; install it explicitly since we test
# `pip/uv install --no-deps` for torchrl itself.
if [[ "$RELEASE" == 0 ]]; then
  uv_pip_install --no-build-isolation --no-deps git+https://github.com/pytorch/tensordict.git
else
  uv_pip_install tensordict
fi

pytest -q test/test_setup.py -vv
