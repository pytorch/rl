#!/usr/bin/env bash

set -euo pipefail

# python3-venv is also needed for the reusable job's cache synchronization.
apt-get update
apt-get install -y curl python3-dev python3-venv libegl1 libgl1-mesa-dri libglfw3
# The checkout is owned by a different user inside the container.
git config --global --add safe.directory '*'
root_dir="$(git rev-parse --show-toplevel)"
cd "$root_dir"

if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv venv --seed --python "$PYTHON_VERSION" "$root_dir/env"
source "$root_dir/env/bin/activate"

# LeRobot 0.6.1 requires torch < 2.12 and torchvision < 0.27.
python -m pip install 'torch==2.11.0+cpu' 'torchvision==0.26.0+cpu' --index-url https://download.pytorch.org/whl/cpu
# setuptools_scm 10 pulls vcs-versioning, which requires packaging >= 26.2;
# LeRobot 0.6.1 requires packaging < 26.
python -m pip install setuptools wheel 'setuptools_scm>=8,<10' 'packaging>=24.2,<26' ninja 'pybind11[global]' cmake pytest pytest-timeout pytest-error-for-skips ipython 'mujoco>=3.8.1,<3.9.0'
python -m pip install -e '.[lerobot]' --no-build-isolation

bash .github/unittest/helpers/assert_torch_tensordict_versions.sh stable
python -c 'import torch, torchvision; assert torch.__version__ == "2.11.0+cpu", torch.__version__; assert torchvision.__version__ == "0.26.0+cpu", torchvision.__version__; assert torch.version.cuda is None, torch.version.cuda'
python -m pip check
python -m pytest test/libs/test_datasets.py -q -k "LeRobotDatasetAdapter or lerobot_adapter" --error-for-skips

# Run the tutorial in this compatible environment instead of the nightly tutorial job.
export TORCHRL_TEST_LEROBOT_TUTORIAL=1
export TORCHRL_TUTORIALS_FAST=1
export LIBGL_ALWAYS_SOFTWARE=1
export EGL_PLATFORM=surfaceless
python -m pytest .github/unittest/tutorials/scripts/test_tutorials.py -q -k lerobot_export
