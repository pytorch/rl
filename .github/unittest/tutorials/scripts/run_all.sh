#!/usr/bin/env bash

set -euxo pipefail
set -v

# ==================================================================================== #
# ================================ Init ============================================== #

export DEBIAN_FRONTEND=noninteractive
export TZ="${TZ:-Etc/UTC}"
ln -snf "/usr/share/zoneinfo/${TZ}" /etc/localtime || true
echo "${TZ}" > /etc/timezone || true

apt-get update
apt-get install -y --no-install-recommends tzdata
dpkg-reconfigure -f noninteractive tzdata || true

apt-get upgrade -y
apt-get install -y vim git wget cmake curl python3-dev python3.12-dev

apt-get install -y libglfw3 libosmesa6 libglew-dev libosmesa6-dev
apt-get install -y libglvnd0 libgl1 libglx0 libglx-mesa0 libegl1 libgles2
apt-get install -y g++ gcc patchelf
apt-get install -y ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev
apt-get install -y libavdevice-dev libavfilter-dev libswresample-dev pkg-config

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# from cudagl docker image
cp $this_dir/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json


# ==================================================================================== #
# ================================ Setup env ========================================= #

# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'
root_dir="$(git rev-parse --show-toplevel)"
env_dir="${root_dir}/venv"

cd "${root_dir}"

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Create venv with uv
printf "* Creating venv with Python ${PYTHON_VERSION}\n"
uv venv --python "${PYTHON_VERSION}" "${env_dir}"
source "${env_dir}/bin/activate"

# Verify CPython
python -c "import sys; assert sys.implementation.name == 'cpython', f'Expected CPython, got {sys.implementation.name}'"

# Set environment variables
export SDL_VIDEODRIVER=dummy
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LAZY_LEGACY_OP=False
export COMPOSITE_LP_AGGREGATE=0
export MAX_IDLE_COUNT=1000
export DISPLAY=:99
export BATCHED_PIPE_TIMEOUT=60
export TOKENIZERS_PARALLELISM=true
export UV_LINK_MODE=copy

# ============================================================================================ #
# ================================ PyTorch =================================================== #

if [[ ${#CU_VERSION} -eq 4 ]]; then
    CUDA_VERSION="${CU_VERSION:2:1}.${CU_VERSION:3:1}"
elif [[ ${#CU_VERSION} -eq 5 ]]; then
    CUDA_VERSION="${CU_VERSION:2:2}.${CU_VERSION:4:1}"
fi
echo "Using CUDA $CUDA_VERSION as determined by CU_VERSION ($CU_VERSION)"

# submodules
git submodule sync && git submodule update --init --recursive

printf "Installing PyTorch with %s\n" "${CU_VERSION}"
if [[ "$TORCH_VERSION" == "nightly" ]]; then
  if [ "${CU_VERSION:-}" == cpu ] ; then
      uv pip install --no-progress --upgrade --pre torch torchvision "numpy==1.26.4" --index-url https://download.pytorch.org/whl/nightly/cpu
  else
      uv pip install --no-progress --upgrade --pre torch torchvision "numpy==1.26.4" --index-url https://download.pytorch.org/whl/nightly/$CU_VERSION
  fi
elif [[ "$TORCH_VERSION" == "stable" ]]; then
  if [ "${CU_VERSION:-}" == cpu ] ; then
      uv pip install --no-progress --upgrade torch torchvision "numpy==1.26.4" --index-url https://download.pytorch.org/whl/cpu
  else
      uv pip install --no-progress --upgrade torch torchvision "numpy==1.26.4" --index-url https://download.pytorch.org/whl/$CU_VERSION
  fi
else
  printf "Failed to install pytorch"
  exit 1
fi

# smoke test
python -c "import functorch"
bash "${root_dir}/.github/unittest/helpers/assert_torch_version.sh" "$TORCH_VERSION"

# ============================================================================================ #
# ================================ TorchCodec ================================================ #

printf "* Installing torchcodec\n"
if [[ "$RELEASE" == 0 ]]; then
  uv pip install --no-progress setuptools ninja packaging "pybind11[global]"
  torchcodec_dir="$(mktemp -d)"
  git clone --depth 1 https://github.com/pytorch/torchcodec.git "$torchcodec_dir"
  python_base="$(python -c 'import sys; print(sys.base_prefix)')"
  CMAKE_PREFIX_PATH="${python_base}${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}" \
    I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1 \
    uv pip install --no-progress --no-build-isolation "$torchcodec_dir"
  rm -rf "$torchcodec_dir"
else
  uv pip install --no-progress "torchcodec>=0.10.0"
fi

# ==================================================================================== #
# ================================ Install dependencies ============================== #

printf "* Installing dependencies\n"

# Install base dependencies after PyTorch so dependencies such as vmas do not
# pull a stable torch wheel before the requested nightly/stable build.
uv pip install --no-progress \
  hypothesis \
  future \
  cloudpickle \
  pyvers \
  packaging \
  pygame \
  "moviepy<2.0.0" \
  tqdm \
  pytest \
  pytest-cov \
  pytest-mock \
  pytest-instafail \
  pytest-rerunfailures \
  pytest-json-report \
  expecttest \
  pybind11 \
  pyyaml \
  scipy \
  psutil \
  hydra-core \
  "imageio==2.26.0" \
  dm_control \
  mujoco \
  av \
  coverage \
  vmas \
  "gymnasium[atari,mujoco]>=1.1.0" \
  tensorboard \
  matplotlib \
  wandb \
  onnxruntime

# ============================================================================================ #
# ================================ TorchRL =================================================== #

# install tensordict
if [[ "$RELEASE" == 0 ]]; then
  uv pip install --no-progress --no-deps git+https://github.com/pytorch/tensordict.git
else
  uv pip install --no-progress --no-deps tensordict
fi

printf "* Installing torchrl\n"
uv pip install --no-progress -e . --no-build-isolation --no-deps

# ==================================================================================== #
# ================================ Run tests ========================================= #

bash ${this_dir}/run_test.sh

# ==================================================================================== #
# ================================ Post-proc ========================================= #

bash ${this_dir}/post_process.sh
