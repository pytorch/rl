#!/usr/bin/env bash

set -euxo pipefail

unset PYTORCH_VERSION

root_dir="$(git rev-parse --show-toplevel)"
env_dir="${root_dir}/env"
source "${env_dir}/bin/activate"
export PATH="$HOME/.local/bin:$PATH"

uv_pip_install() {
  uv pip install --no-progress --python "${env_dir}/bin/python" "$@"
}

if [ "${CU_VERSION:-}" == cpu ] ; then
  torch_index="https://download.pytorch.org/whl/nightly/cpu"
  stable_torch_index="https://download.pytorch.org/whl/cpu"
else
  torch_index="https://download.pytorch.org/whl/nightly/${CU_VERSION}"
  stable_torch_index="https://download.pytorch.org/whl/${CU_VERSION}"
fi

git submodule sync && git submodule update --init --recursive

uv_pip_install \
  cloudpickle \
  coverage \
  expecttest \
  future \
  hydra-core \
  hypothesis \
  importlib_metadata \
  orjson \
  packaging \
  psutil \
  "pyvers>=0.2.3,<0.3.0" \
  pybind11[global] \
  pytest \
  pytest-asyncio \
  pytest-cov \
  pytest-error-for-skips \
  pytest-instafail \
  pytest-json-report \
  pytest-mock \
  pytest-rerunfailures \
  pytest-timeout \
  pyyaml \
  scipy \
  setuptools \
  wheel

if [[ "$TORCH_VERSION" == "nightly" ]]; then
  uv_pip_install --upgrade --pre torch --index-url "${torch_index}"
elif [[ "$TORCH_VERSION" == "stable" ]]; then
  uv_pip_install --upgrade torch --index-url "${stable_torch_index}"
else
  echo "Failed to install pytorch"
  exit 1
fi

# Install TensorDict before TorchRL. Nightly CI validates against TensorDict main.
if [[ "$RELEASE" == 0 ]]; then
  uv_pip_install --no-build-isolation --no-deps git+https://github.com/pytorch/tensordict.git
else
  uv_pip_install --no-deps tensordict
fi

uv_pip_install -e . --no-build-isolation --no-deps

# LIBERO is source-only. Its root is a namespace-package parent; keep it on
# PYTHONPATH at test time instead of relying on the upstream editable install.
libero_dir="${root_dir}/libero-src"
rm -rf "${libero_dir}"
git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git "${libero_dir}"

# robosuite 1.4.0 calls the pre-3.10 mj_fullM signature.
uv_pip_install \
  "bddl==1.0.1" \
  easydict \
  "gym==0.25.2" \
  h5py \
  imageio \
  matplotlib \
  "mujoco<3.10.0" \
  "numpy<2" \
  opencv-python \
  "robosuite==1.4.0" \
  termcolor \
  tqdm

timeout 120s python -c "import functorch; import tensordict; import torchrl"

export LIBERO_CONFIG_PATH="${root_dir}/.libero-ci"
timeout 120s env PYTHONPATH="${libero_dir}:${PYTHONPATH:-}" python -c "from torchrl.envs.libs.libero import _ensure_libero_config; _ensure_libero_config()"
timeout 120s env PYTHONPATH="${libero_dir}:${PYTHONPATH:-}" python -c "from libero.libero import benchmark; from libero.libero.envs import OffScreenRenderEnv; print(sorted(benchmark.get_benchmark_dict()))"
