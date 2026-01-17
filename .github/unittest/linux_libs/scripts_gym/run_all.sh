#!/usr/bin/env bash

# Consolidated script for gym tests using uv instead of conda

set -e
set -v

# 1. Install system dependencies FIRST (before using git)
printf "* Installing system dependencies\n"
export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y --no-install-recommends git wget gcc g++ curl software-properties-common
apt-get install -y --no-install-recommends libglfw3 libgl1-mesa-glx libosmesa6 libglew-dev libsdl2-dev libsdl2-2.0-0
apt-get install -y --no-install-recommends libglvnd0 libgl1 libglx0 libegl1 libgles2 xvfb libegl-dev libx11-dev freeglut3-dev
apt-get install -y --no-install-recommends librhash0 x11proto-dev cmake

# Install Python 3.10 (Ubuntu 22.04 has Python 3.10 as default)
apt-get install -y --no-install-recommends python3 python3-dev python3-venv python3-pip

# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'

# Now we can use git
this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
root_dir="$(git rev-parse --show-toplevel)"
env_dir="${root_dir}/.venv"

cd "${root_dir}"

# 2. Install uv
printf "* Installing uv\n"
curl -LsSf https://astral.sh/uv/install.sh | sh
# Ensure uv is in PATH
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# 3. Create virtual environment with uv
printf "* Creating virtual environment with Python ${PYTHON_VERSION:-3.10}\n"
uv venv "${env_dir}" --python "${PYTHON_VERSION:-3.10}"
source "${env_dir}/bin/activate"

# 4. Install base dependencies
printf "* Installing base dependencies\n"
uv pip install --upgrade pip setuptools==65.3.0 wheel
uv pip install charset-normalizer

# 5. Install PyTorch
printf "* Installing PyTorch with ${CU_VERSION}\n"
if [ "${CU_VERSION:-}" == "cpu" ]; then
    uv pip install torch==2.0 torchvision==0.15 --index-url https://download.pytorch.org/whl/cpu
else
    if [[ ${#CU_VERSION} -eq 4 ]]; then
        CUDA_VERSION="${CU_VERSION:2:1}.${CU_VERSION:3:1}"
    elif [[ ${#CU_VERSION} -eq 5 ]]; then
        CUDA_VERSION="${CU_VERSION:2:2}.${CU_VERSION:4:1}"
    fi
    echo "Using CUDA $CUDA_VERSION as determined by CU_VERSION ($CU_VERSION)"
    uv pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
    uv pip install numpy==1.26
fi

# 6. Install pybind11 and ninja (needed for building tensordict and torchrl C++ extensions)
printf "* Installing pybind11 and ninja\n"
uv pip install "pybind11[global]>=2.13" ninja

# Help CMake find pybind11 when building from source
pybind11_DIR="$(python -m pybind11 --cmakedir)"
export pybind11_DIR

# 7. Install tensordict
printf "* Installing tensordict\n"
if [[ "$RELEASE" == 0 ]]; then
    # Install tensordict dependencies (since we use --no-deps)
    uv pip install cloudpickle packaging importlib_metadata orjson "pyvers>=0.1.0,<0.2.0"
    uv pip install --no-build-isolation --no-deps git+https://github.com/pytorch/tensordict.git
else
    uv pip install --no-deps tensordict
fi

# Smoke test tensordict
python -c "import tensordict"

# 8. Install torchrl
printf "* Installing torchrl\n"
git submodule sync && git submodule update --init --recursive
uv pip install -e . --no-build-isolation --no-deps

# Smoke test torchrl
python -c "import torchrl"

# 9. Setup mujoco
printf "* Setting up mujoco\n"
if [ ! -d "mujoco-py" ]; then
    git clone https://github.com/vmoens/mujoco-py.git
    cd mujoco-py
    git checkout aws_fix2
    mkdir -p mujoco_py/binaries/linux \
        && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
        && tar -xf mujoco.tar.gz -C mujoco_py/binaries/linux \
        && rm mujoco.tar.gz
    wget https://pytorch.s3.amazonaws.com/torchrl/github-artifacts/mjkey.txt
    cp mjkey.txt mujoco_py/binaries/
    cd ..
fi

# Install mujoco-py
printf "* Installing mujoco-py\n"
cd mujoco-py
uv pip install -e .
cd "${root_dir}"

# 10. Install base test dependencies (including gym[atari]==0.13 for Atari ROMs)
printf "* Installing test dependencies\n"
uv pip install \
    protobuf \
    'gym[atari]==0.13' \
    hypothesis \
    future \
    cloudpickle \
    pygame \
    "moviepy<2.0.0" \
    tqdm \
    pytest \
    pytest-cov \
    pytest-mock \
    pytest-instafail \
    pytest-rerunfailures \
    pytest-error-for-skips \
    pytest-asyncio \
    expecttest \
    pyyaml \
    scipy \
    hydra-core \
    patchelf \
    pyopengl==3.1.0 \
    coverage \
    ale_py

# 11. Setup Atari ROMs
printf "* Setting up Atari ROMs\n"
if [ ! -d "Roms" ]; then
    wget https://www.rarlab.com/rar/rarlinux-x64-5.7.1.tar.gz --no-check-certificate
    tar -xzvf rarlinux-x64-5.7.1.tar.gz
    mkdir -p Roms
    wget http://www.atarimania.com/roms/Roms.rar
    ./rar/unrar e Roms.rar ./Roms -y
fi
python -m atari_py.import_roms Roms

# 12. Set environment variables
export MUJOCO_GL=egl
export MAX_IDLE_COUNT=1000
export SDL_VIDEODRIVER=dummy
export DISPLAY=:99
export PYOPENGL_PLATFORM=egl
export MUJOCO_PY_MJKEY_PATH=${root_dir}/mujoco-py/mujoco_py/binaries/mjkey.txt
export MUJOCO_PY_MUJOCO_PATH=${root_dir}/mujoco-py/mujoco_py/binaries/linux/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${root_dir}/mujoco-py/mujoco_py/binaries/linux/mujoco210/bin
export TOKENIZERS_PARALLELISM=true
export PYTORCH_TEST_WITH_SLOW='1'
export LAZY_LEGACY_OP=False
export MKL_THREADING_LAYER=GNU

# 13. Start Xvfb for display
printf "* Starting Xvfb\n"
unset LD_PRELOAD
Xvfb :99 -screen 0 1400x900x24 > /dev/null 2>&1 &

# 14. Function to run tests
run_tests() {
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${env_dir}/lib
    
    python -m torch.utils.collect_env
    python .github/unittest/helpers/coverage_run_parallel.py -m pytest test/smoke_test.py -v --durations 200
    python .github/unittest/helpers/coverage_run_parallel.py -m pytest test/smoke_test_deps.py -v --durations 200 -k 'test_gym'
    python .github/unittest/helpers/coverage_run_parallel.py -m pytest test/test_libs.py --instafail -v --durations 200 -k "gym and not isaac" --mp_fork
    
    coverage combine -q
    coverage xml -i
}

# 15. Run tests for different gym versions
printf "* Running tests for different gym versions\n"

# Test gym 0.13 (already installed)
printf "* Testing gym 0.13\n"
run_tests
uv pip uninstall gym atari-py

# Test gym 0.19 (broken metadata, needs pip<24.1)
printf "* Testing gym 0.19\n"
pip install "pip<24.1" setuptools==65.3.0 wheel==0.38.4
pip install gym==0.19
run_tests
pip uninstall -y gym wheel
pip install --upgrade pip setuptools wheel  # restore latest versions

# Test gym 0.20 (also needs older pip for metadata issues)
printf "* Testing gym 0.20\n"
pip install "pip<24.1" setuptools==65.3.0 wheel==0.38.4
pip install 'gym[atari]==0.20'
pip install 'ale-py==0.7.4'
run_tests
pip uninstall -y gym ale-py wheel
pip install --upgrade pip setuptools wheel  # restore latest versions

# Test gym 0.25 (needs both mujoco-py for env and mujoco for rendering)
printf "* Testing gym 0.25\n"
# gym 0.25 requires mujoco-py for HalfCheetah-v4 AND mujoco for rendering
uv pip install 'numpy>=1.21,<1.24'  # gym 0.25 needs numpy<1.24 for AsyncVectorEnv deepcopy compatibility
uv pip install 'gym[atari]==0.25' mujoco
run_tests
uv pip uninstall gym

# Test gym 0.26 (uses new mujoco bindings for HalfCheetah-v4)
printf "* Testing gym 0.26\n"
# Uninstall mujoco-py and switch to new mujoco package for gym 0.26+
uv pip uninstall mujoco-py
unset MUJOCO_PY_MJKEY_PATH MUJOCO_PY_MUJOCO_PATH
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | sed 's|[^:]*mujoco210[^:]*:*||g')
uv pip install 'numpy>=1.21,<1.24'  # gym 0.26 needs numpy<1.24 for AsyncVectorEnv deepcopy compatibility
uv pip install 'gym[atari,accept-rom-license]==0.26' mujoco
uv pip install gym-super-mario-bros
run_tests
uv pip uninstall gym gym-super-mario-bros mujoco

# Test gymnasium 0.27 and 0.28
for GYM_VERSION in '0.27' '0.28'; do
    printf "* Testing gymnasium ${GYM_VERSION}\n"
    uv pip install "gymnasium[atari,ale-py]==${GYM_VERSION}"
    run_tests
    uv pip uninstall gymnasium ale-py
done

# Test gymnasium >=1.1.0
printf "* Testing gymnasium >=1.1.0\n"
uv pip install 'gymnasium[ale-py,atari]>=1.1.0' mo-gymnasium gymnasium-robotics
run_tests
uv pip uninstall gymnasium mo-gymnasium gymnasium-robotics ale-py

# Test latest gymnasium
printf "* Testing latest gymnasium\n"
uv pip install 'gymnasium[ale-py,atari]>=1.1.0' mo-gymnasium gymnasium-robotics
run_tests

printf "* All tests completed\n"
