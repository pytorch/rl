#!/usr/bin/env bash

set -euxo pipefail

export DEBIAN_FRONTEND=noninteractive
export TZ="${TZ:-Etc/UTC}"
ln -snf "/usr/share/zoneinfo/${TZ}" /etc/localtime || true
echo "${TZ}" > /etc/timezone || true

apt-get update
apt-get install -y --no-install-recommends tzdata
apt-get upgrade -y
apt-get install -y --no-install-recommends \
  cmake \
  curl \
  ffmpeg \
  g++ \
  gcc \
  git \
  libavcodec-dev \
  libavdevice-dev \
  libavfilter-dev \
  libavformat-dev \
  libavutil-dev \
  libegl1 \
  libglew-dev \
  libgles2 \
  libgl1 \
  libglfw3 \
  libglvnd0 \
  libglx0 \
  libosmesa6-dev \
  libswresample-dev \
  libswscale-dev \
  patchelf \
  pkg-config \
  python3-dev \
  unzip \
  wget \
  xvfb
apt-get upgrade -y libstdc++6

# Ensure EGL discovers the NVIDIA vendor library in the CUDA runner image.
mkdir -p /usr/share/glvnd/egl_vendor.d
cp .github/unittest/linux/scripts/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json || true

# Avoid error: "fatal: unsafe repository".
git config --global --add safe.directory '*'

root_dir="$(git rev-parse --show-toplevel)"
env_dir="${root_dir}/env"

cd "${root_dir}"

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

rm -rf "${env_dir}"
uv venv --python "${PYTHON_VERSION}" "${env_dir}"
