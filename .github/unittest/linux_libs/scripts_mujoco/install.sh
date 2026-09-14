#!/usr/bin/env bash

unset PYTORCH_VERSION
# For unittest, nightly PyTorch is used as the following section,
# so no need to set PYTORCH_VERSION.
# In fact, keeping PYTORCH_VERSION forces us to hardcode PyTorch version in config.

set -euxo pipefail

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

if [ "${CU_VERSION:-}" == cpu ] ; then
    version="cpu"
else
    if [[ ${#CU_VERSION} -eq 4 ]]; then
        CUDA_VERSION="${CU_VERSION:2:1}.${CU_VERSION:3:1}"
    elif [[ ${#CU_VERSION} -eq 5 ]]; then
        CUDA_VERSION="${CU_VERSION:2:2}.${CU_VERSION:4:1}"
    fi
    echo "Using CUDA $CUDA_VERSION as determined by CU_VERSION ($CU_VERSION)"
    version="$(python -c "print('.'.join(\"${CUDA_VERSION}\".split('.')[:2]))")"
fi

# submodules
git submodule sync && git submodule update --init --recursive

printf "Installing PyTorch with cu128\n"
if [[ "$TORCH_VERSION" == "nightly" ]]; then
  if [ "${CU_VERSION:-}" == cpu ] ; then
      pip3 install --pre --force-reinstall torch --index-url https://download.pytorch.org/whl/nightly/cpu -U
  else
      pip3 install --pre --force-reinstall torch --index-url https://download.pytorch.org/whl/nightly/cu128 -U
  fi
elif [[ "$TORCH_VERSION" == "stable" ]]; then
    if [ "${CU_VERSION:-}" == cpu ] ; then
      pip3 install --force-reinstall torch --index-url https://download.pytorch.org/whl/cpu -U
  else
      pip3 install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
  fi
else
  printf "Failed to install pytorch"
  exit 1
fi


# Keep the native and MJX versions paired. The mujoco-torch revision includes
# the sparse mass-matrix batching and multi-mesh rendering fixes (#88, #89),
# required by the football scene but absent from the 0.2.0 release.
pip install mujoco==3.7.0 mujoco-mjx==3.7.0 'jax[cuda12]>=0.7.0,<0.11' --progress-bar off
pip install 'mujoco-torch @ https://github.com/vmoens/mujoco-torch/archive/08ec29fbadf18fc51f0c52ee6836a3788f370353.zip' --no-deps --progress-bar off

# install tensordict
pip install git+https://github.com/pytorch/tensordict.git --progress-bar off

# smoke test
python -c "import functorch;import tensordict"

printf "* Installing torchrl\n"
# Keep the source TensorDict build: its development version can sort below the
# release dependency bound even when it contains newer compiler fixes.
pip install 'hoptorch>=0.1.4' --no-deps --progress-bar off
python -m pip install -e . --no-build-isolation --no-deps

# smoke test
python -c "import torchrl"
python -c "import mujoco; import mujoco.mjx; import mujoco_torch; print('mujoco', mujoco.__version__, 'mujoco-torch ok')"
