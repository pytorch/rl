#!/usr/bin/env bash

set -euxo pipefail

export DEBIAN_FRONTEND=noninteractive
export TZ="${TZ:-Etc/UTC}"
ln -snf "/usr/share/zoneinfo/${TZ}" /etc/localtime || true
echo "${TZ}" >/etc/timezone || true

apt-get update
apt-get install -y --no-install-recommends \
  curl ffmpeg g++ gcc git libegl1 libgl1 libgles2 libglfw3 libglvnd0 \
  libglx-mesa0 libglew-dev libosmesa6 libosmesa6-dev python3-dev tzdata

this_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
root_dir="$(cd "${this_dir}/../../../.." >/dev/null 2>&1 && pwd)"
env_dir="${root_dir}/venv"

cp "${root_dir}/.github/unittest/tutorials/scripts/10_nvidia.json" \
  /usr/share/glvnd/egl_vendor.d/10_nvidia.json
git config --global --add safe.directory '*'
cd "${root_dir}"

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="${HOME}/.local/bin:${PATH}"
uv venv --python "${PYTHON_VERSION}" "${env_dir}"
# shellcheck disable=SC1091
source "${env_dir}/bin/activate"

uv_pip_install() {
  uv pip install --no-progress --python "${env_dir}/bin/python" "$@"
}

torch_index_root="https://download.pytorch.org/whl/nightly"
if [[ "${TORCH_VERSION}" == stable ]]; then
  torch_index_root="https://download.pytorch.org/whl"
fi
if [[ "${CU_VERSION}" == cpu ]]; then
  torch_index="${torch_index_root}/cpu"
else
  torch_index="${torch_index_root}/${CU_VERSION}"
fi
if [[ "${TORCH_VERSION}" == stable ]]; then
  uv_pip_install --upgrade torch torchvision --index-url "${torch_index}"
else
  uv_pip_install --upgrade --pre torch torchvision --index-url "${torch_index}"
fi
bash "${root_dir}/.github/unittest/helpers/assert_torch_version.sh" "${TORCH_VERSION}"

uv_pip_install \
  av cloudpickle configargparse coverage datasets "dm_control==1.0.39" future h5py \
  "gym==0.26.2" "gymnasium[mujoco]>=1.1.0" \
  "hydra-core<1.4" imageio matplotlib moviepy packaging psutil pygame \
  "pybind11[global]" pytest pytest-cov pytest-instafail pytest-json-report \
  pytest-timeout pyyaml "pyvers>=0.2.3" ray scipy tensorboard tqdm \
  transformers vmas wandb
uv_pip_install "mujoco==3.7.0"
uv_pip_install --no-deps "mujoco-torch==0.2.0"

uv_pip_install ninja setuptools
if [[ "${RELEASE}" == 0 ]]; then
  uv_pip_install --no-build-isolation --no-deps git+https://github.com/pytorch/tensordict.git
else
  uv_pip_install --no-deps tensordict
fi
uv_pip_install -e . --no-build-isolation --no-deps

menagerie_dir="${root_dir}/.mujoco_menagerie"
git clone --depth=1 --filter=blob:none --sparse \
  https://github.com/google-deepmind/mujoco_menagerie.git "${menagerie_dir}"
git -C "${menagerie_dir}" sparse-checkout set universal_robots_ur5e robotiq_2f85
export TORCHRL_MUJOCO_MENAGERIE_PATH="${menagerie_dir}"

export BATCHED_PIPE_TIMEOUT=60
export COMPOSITE_LP_AGGREGATE=0
export CUDA_LAUNCH_BLOCKING=1
export DISPLAY=:99
export LAZY_LEGACY_OP=False
export MKL_THREADING_LAYER=GNU
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export SDL_VIDEODRIVER=dummy
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled

bash "${root_dir}/.github/unittest/helpers/assert_torch_tensordict_versions.sh" "${TORCH_VERSION}"
bash "${this_dir}/run_test.sh"
