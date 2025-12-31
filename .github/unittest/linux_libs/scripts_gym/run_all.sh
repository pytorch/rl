#!/usr/bin/env bash

# Consolidated script for gym tests using uv instead of conda

set -e
set -v

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
root_dir="$(git rev-parse --show-toplevel)"
env_dir="${root_dir}/.venv"

cd "${root_dir}"

# 1. Install system dependencies
printf "* Installing system dependencies\n"
apt-get update && apt-get install -y git wget gcc g++
apt-get install -y libglfw3 libgl1-mesa-glx libosmesa6 libglew-dev libsdl2-dev libsdl2-2.0-0
apt-get install -y libglvnd0 libgl1 libglx0 libegl1 libgles2 xvfb libegl-dev libx11-dev freeglut3-dev
apt-get install -y librhash0 libx11-dev x11proto-dev
apt-get install -y python3.9 python3.9-dev python3.9-venv

# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'

# 2. Install uv
if ! command -v uv &> /dev/null; then
    printf "* Installing uv\n"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi
# Ensure uv is in PATH
export PATH="$HOME/.cargo/bin:$PATH"

# 3. Setup mujoco
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

# 4. Setup Atari ROMs
printf "* Setting up Atari ROMs\n"
if [ ! -d "Roms" ]; then
    wget https://www.rarlab.com/rar/rarlinux-x64-5.7.1.tar.gz --no-check-certificate
    tar -xzvf rarlinux-x64-5.7.1.tar.gz
    mkdir Roms
    wget http://www.atarimania.com/roms/Roms.rar
    ./rar/unrar e Roms.rar ./Roms -y
fi

# 5. Create virtual environment with uv
printf "* Creating virtual environment with Python ${PYTHON_VERSION:-3.9}\n"
uv venv "${env_dir}" --python "${PYTHON_VERSION:-3.9}"
source "${env_dir}/bin/activate"

# 6. Install base dependencies
printf "* Installing base dependencies\n"
uv pip install --upgrade pip setuptools==65.3.0 wheel
uv pip install charset-normalizer

# 7. Install PyTorch
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

# 8. Install tensordict
printf "* Installing tensordict\n"
if [[ "$RELEASE" == 0 ]]; then
    # Install cmake for building tensordict
    apt-get install -y cmake
    uv pip install "pybind11[global]"
    uv pip install git+https://github.com/pytorch/tensordict.git --no-deps --ignore-requires-python
else
    uv pip install tensordict --no-deps --ignore-requires-python
fi

# Smoke test tensordict
python -c "import tensordict"

# 9. Install torchrl
printf "* Installing torchrl\n"
git submodule sync && git submodule update --init --recursive
uv pip install -e . --no-build-isolation --no-deps --ignore-requires-python

# Smoke test torchrl
python -c "import torchrl"

# 10. Install mujoco-py
printf "* Installing mujoco-py\n"
cd mujoco-py
uv pip install -e .
cd ..

# Import Atari ROMs
python -m atari_py.import_roms Roms

# 11. Set environment variables
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

# 12. Install base test dependencies
printf "* Installing test dependencies\n"
uv pip install \
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
    pyopengl==3.1.0

# 13. Start Xvfb for display
printf "* Starting Xvfb\n"
unset LD_PRELOAD
export DISPLAY=:99
Xvfb :99 -screen 0 1400x900x24 > /dev/null 2>&1 &

# 14. Run tests for different gym versions
printf "* Running tests for different gym versions\n"

# Function to run tests
run_tests() {
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$env_dir/lib
    
    python .github/unittest/helpers/coverage_run_parallel.py -m pytest test/smoke_test.py -v --durations 200
    python .github/unittest/helpers/coverage_run_parallel.py -m pytest test/smoke_test_deps.py -v --durations 200 -k 'test_gym'
    python .github/unittest/helpers/coverage_run_parallel.py -m pytest test/test_libs.py --instafail -v --durations 200 -k "gym and not isaac" --error-for-skips --mp_fork
    
    coverage combine -q
    coverage xml -i
}

# Test gym 0.13
printf "* Testing gym 0.13\n"
uv pip install 'gym[atari]==0.13'
run_tests
uv pip uninstall -y gym

# Test gym 0.19 (broken, install without dependencies)
printf "* Testing gym 0.19\n"
uv pip install wheel==0.38.4
uv pip install "pip<24.1"
uv pip install gym==0.19
run_tests
uv pip uninstall -y gym wheel

# Test gym 0.20
printf "* Testing gym 0.20\n"
uv pip install wheel==0.38.4
uv pip install "pip<24.1"
uv pip install 'gym[atari]==0.20'
uv pip install ale-py==0.7
run_tests
uv pip uninstall -y gym ale-py wheel

# Test gym 0.25
printf "* Testing gym 0.25\n"
uv pip install 'gym[atari]==0.25'
uv pip install --upgrade pip
run_tests
uv pip uninstall -y gym

# Test gym 0.26
printf "* Testing gym 0.26\n"
uv pip install 'gym[atari,accept-rom-license]==0.26'
uv pip install gym-super-mario-bros
run_tests
uv pip uninstall -y gym gym-super-mario-bros

# Test gymnasium 0.27 and 0.28
for GYM_VERSION in '0.27' '0.28'; do
    printf "* Testing gymnasium ${GYM_VERSION}\n"
    uv pip install "gymnasium[atari,ale-py]==${GYM_VERSION}"
    run_tests
    uv pip uninstall -y gymnasium
done

# Test latest gymnasium (prev)
printf "* Testing previous gymnasium\n"
uv pip install 'gymnasium[ale-py,atari]>=1.1.0' mo-gymnasium gymnasium-robotics -U
run_tests
uv pip uninstall -y gymnasium mo-gymnasium gymnasium-robotics

# Test latest gymnasium
printf "* Testing latest gymnasium\n"
uv pip install 'gymnasium[ale-py,atari]>=1.1.0' mo-gymnasium gymnasium-robotics -U
run_tests

printf "* All tests completed\n"
