#!/usr/bin/env bash

unset PYTORCH_VERSION
# For unittest, nightly PyTorch is used as the following section,
# so no need to set PYTORCH_VERSION.
# In fact, keeping PYTORCH_VERSION forces us to hardcode PyTorch version in config.
apt-get update && apt-get install -y git wget libglew-dev libx11-dev x11proto-dev g++ gcc libosmesa6-dev

set -e
set -v

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

#apt-get update -y && apt-get install git wget gcc g++ -y

if [ "${CU_VERSION:-}" == cpu ] ; then
    cudatoolkit="cpuonly"
    version="cpu"
else
    if [[ ${#CU_VERSION} -eq 4 ]]; then
        CUDA_VERSION="${CU_VERSION:2:1}.${CU_VERSION:3:1}"
    elif [[ ${#CU_VERSION} -eq 5 ]]; then
        CUDA_VERSION="${CU_VERSION:2:2}.${CU_VERSION:4:1}"
    fi
    echo "Using CUDA $CUDA_VERSION as determined by CU_VERSION ($CU_VERSION)"
    version="$(python -c "print('.'.join(\"${CUDA_VERSION}\".split('.')[:2]))")"
    cudatoolkit="cudatoolkit=${version}"
fi

case "$(uname -s)" in
    Darwin*) os=MacOSX;;
    *) os=Linux
esac

# submodules
git submodule sync && git submodule update --init --recursive

printf "Installing PyTorch with %s\n" "${CU_VERSION}"
if [ "${CU_VERSION:-}" == cpu ] ; then
    # conda install -y pytorch torchvision cpuonly -c pytorch-nightly
    # use pip to install pytorch as conda can frequently pick older release
#    conda install -y pytorch cpuonly -c pytorch-nightly
    pip3 install --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cpu --force-reinstall
else
    pip3 install --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cu116 --force-reinstall
fi

# install tensordict
pip install git+https://github.com/pytorch-labs/tensordict.git

# smoke test
python -c "import tensordict"

printf "* Installing torchrl\n"
python setup.py develop
python -c "import torchrl"

# Extracted from run_test.sh to run once.

export PYTORCH_TEST_WITH_SLOW='1'
python -m torch.utils.collect_env
# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'

root_dir="$(git rev-parse --show-toplevel)"
env_dir="${root_dir}/env"
lib_dir="${env_dir}/lib"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$lib_dir
export MKL_THREADING_LAYER=GNU

python .github/unittest/helpers/coverage_run_parallel.py -m pytest test/smoke_test.py -v --durations 20

# let's make sure we have a GPU at our disposal
python -c """
import torch
devcount = torch.cuda.device_count()
assert devcount
print('device count', devcount)
"""

echo $MUJOCO_GL
echo $sim_backend

sim_backend=MUJOCO MUJOCO_GL=egl python .github/unittest/helpers/coverage_run_parallel.py -m pytest test/test_libs.py --instafail -v --durations 20 -k "robohive" --error-for-skips
coverage combine
coverage xml -i
