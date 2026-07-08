#!/usr/bin/env bash

set -euxo pipefail
set -v
set -e

# =============================================================================== #
# ================================ Init ========================================= #


if [[ $OSTYPE != 'darwin'* ]]; then
  # Prevent interactive prompts (notably tzdata) in CI.
  export DEBIAN_FRONTEND=noninteractive
  export TZ="${TZ:-Etc/UTC}"
  ln -snf "/usr/share/zoneinfo/${TZ}" /etc/localtime || true
  echo "${TZ}" > /etc/timezone || true

  apt-get update
  apt-get install -y --no-install-recommends tzdata
  dpkg-reconfigure -f noninteractive tzdata || true

  # Single install pass, no recommends: a blanket `apt-get upgrade` /
  # `dist-upgrade` of the throwaway container costs minutes per job and
  # provides nothing the tests need.
  apt-get install -y --no-install-recommends \
    vim git wget cmake \
    libglfw3 libosmesa6 libglew-dev \
    libglvnd0 libgl1 libglx0 libglx-mesa0 libegl1 libgles2 xvfb ffmpeg

  if [ "${CU_VERSION:-}" == cpu ] ; then
    # solves version `GLIBCXX_3.4.29' not found for tensorboard; upgrade
    # libstdc++6 specifically instead of dist-upgrading the whole image.
    apt-get install -y --only-upgrade libstdc++6
  else
    apt-get install -y --no-install-recommends g++ gcc
  fi

fi

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
if [[ $OSTYPE != 'darwin'* ]]; then
  # from cudagl docker image
  cp $this_dir/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
fi


# ==================================================================================== #
# ================================ Setup env ========================================= #

# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'
root_dir="$(git rev-parse --show-toplevel)"
conda_dir="${root_dir}/conda"
env_dir="${root_dir}/env"
lib_dir="${env_dir}/lib"

cd "${root_dir}"

case "$(uname -s)" in
    Darwin*) os=MacOSX;;
    *) os=Linux
esac

# 1. Install conda at ./conda
if [ ! -d "${conda_dir}" ]; then
    printf "* Installing conda\n"
    wget -O miniconda.sh "http://repo.continuum.io/miniconda/Miniconda3-latest-${os}-x86_64.sh"
    bash ./miniconda.sh -b -f -p "${conda_dir}"
fi
eval "$(${conda_dir}/bin/conda shell.bash hook)"

# 2. Create test environment at ./env
printf "python: ${PYTHON_VERSION}\n"
if [ ! -d "${env_dir}" ]; then
    printf "* Creating a test environment\n"
    conda create --prefix "${env_dir}" -y python="$PYTHON_VERSION"
fi
conda activate "${env_dir}"

# 3. Install Conda dependencies
printf "* Installing dependencies (except PyTorch)\n"
echo "  - python=${PYTHON_VERSION}" >> "${this_dir}/environment.yml"
cat "${this_dir}/environment.yml"

pip3 install pip --upgrade

conda env update --file "${this_dir}/environment.yml" --prune

# ============================================================================================ #
# ================================ PyTorch & TorchRL ========================================= #

unset PYTORCH_VERSION

if [ "${CU_VERSION:-}" == cpu ] ; then
    version="cpu"
    echo "Using cpu build"
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

printf "Installing PyTorch with %s\n" "${CU_VERSION}"
if [[ "$TORCH_VERSION" == "nightly" ]]; then
  if [ "${CU_VERSION:-}" == cpu ] ; then
      pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu -U
  else
      pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/$CU_VERSION -U
  fi
elif [[ "$TORCH_VERSION" == "stable" ]]; then
    if [ "${CU_VERSION:-}" == cpu ] ; then
      pip3 install torch --index-url https://download.pytorch.org/whl/cpu -U
  else
      pip3 install torch --index-url https://download.pytorch.org/whl/$CU_VERSION -U
  fi
else
  printf "Failed to install pytorch"
  exit 1
fi

# smoke test
python -c "import functorch"

## install snapshot
#if [[ "$TORCH_VERSION" == "nightly" ]]; then
#  pip3 install git+https://github.com/pytorch/torchsnapshot
#else
#  pip3 install torchsnapshot
#fi

# install tensordict
pip3 install cloudpickle packaging importlib_metadata numpy orjson "pyvers>=0.2.0,<0.3.0"
if [[ "$RELEASE" == 0 ]]; then
  pip3 install --no-deps git+https://github.com/pytorch/tensordict.git
else
  pip3 install --no-deps tensordict
fi

printf "* Installing hoptorch\n"
python -m pip install "hoptorch>=0.1.1"

printf "* Installing torchrl\n"
python -m pip install -e . --no-build-isolation --no-deps

# smoke test
python -c "import torchrl"

# ==================================================================================== #
# ================================ Run tests ========================================= #


# find libstdc
STDC_LOC=$(find conda/ -name "libstdc++.so.6" | head -1)

export PYTORCH_TEST_WITH_SLOW='1'
export LAZY_LEGACY_OP=False
python -m torch.utils.collect_env

bash "${root_dir}/.github/unittest/helpers/assert_torch_version.sh" "$TORCH_VERSION"
bash "${root_dir}/.github/unittest/helpers/assert_torch_tensordict_versions.sh" "$TORCH_VERSION"
# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'
root_dir="$(git rev-parse --show-toplevel)"

export MKL_THREADING_LAYER=GNU
export CKPT_BACKEND=torch
export MAX_IDLE_COUNT=100
export BATCHED_PIPE_TIMEOUT=60

COMMON_IGNORES="--ignore test/test_rlhf.py --ignore test/test_distributed.py --ignore test/llm"
COMMON_ARGS="--instafail --durations 200 -vv --capture no --mp_fork_if_no_cuda"

# Track test failures but keep going, so coverage is still combined and
# uploaded; the script exits with this status at the end.
EXIT_STATUS=0

if [ "${TORCHRL_XDIST:-1}" = "1" ]; then
  # Coverage for pytest-xdist workers: execnet workers are plain
  # subprocesses, not multiprocessing children, so coverage's
  # concurrency=multiprocessing does not see them. The .pth calls
  # coverage.process_startup(), a no-op unless COVERAGE_PROCESS_START is set
  # (coverage_run_parallel.py exports it for its subprocess tree).
  python - <<'PY'
import sysconfig
from pathlib import Path

pth = Path(sysconfig.get_paths()["purelib"]) / "coverage_process_startup.pth"
pth.write_text("import coverage; coverage.process_startup()\n")
print(f"Wrote {pth}")
PY

  # Three invocations covering the same union of tests as the single serial
  # run below, without overlap:
  # 1. Parallel bulk under pytest-xdist: everything that neither requires the
  #    GPU nor spawns its own worker processes. OMP/MKL are pinned to one
  #    thread per worker (N workers on an N-core box otherwise serialize on
  #    thread contention) and the per-test timeout is raised: tests that take
  #    ~60s alone can legitimately exceed 120s on a fully loaded machine.
  # 2. The process-spawning set (test/envs, test_collectors.py,
  #    test/services), serial: these spawn their own worker processes or Ray
  #    clusters and flake under a fully loaded machine.
  # 3. GPU-marked tests, serial: xdist workers sharing one device could
  #    oversubscribe GPU memory.
  echo "Running parallel bulk (not gpu, without test/envs, test_collectors.py and test/services)"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python .github/unittest/helpers/coverage_run_parallel.py -m pytest test \
    ${COMMON_IGNORES} \
    --ignore test/envs \
    --ignore test/test_collectors.py \
    --ignore test/services \
    -m "not gpu" \
    -n "${TORCHRL_XDIST_WORKERS:-auto}" --dist worksteal \
    ${COMMON_ARGS} --timeout=300 || EXIT_STATUS=$?
  echo "Running process-spawning tests serially (test/envs, test_collectors.py, test/services)"
  python .github/unittest/helpers/coverage_run_parallel.py -m pytest test/envs test/test_collectors.py test/services \
    ${COMMON_ARGS} --timeout=120 || EXIT_STATUS=$?
  echo "Running gpu-marked tests serially"
  python .github/unittest/helpers/coverage_run_parallel.py -m pytest test \
    ${COMMON_IGNORES} \
    --ignore test/envs \
    --ignore test/test_collectors.py \
    --ignore test/services \
    -m gpu \
    ${COMMON_ARGS} --timeout=120 || EXIT_STATUS=$?
  if [ $EXIT_STATUS -ne 0 ]; then
    echo "Some tests failed with exit status $EXIT_STATUS"
  fi
else
  python .github/unittest/helpers/coverage_run_parallel.py -m pytest test \
    ${COMMON_IGNORES} \
    ${COMMON_ARGS} --timeout=120 || EXIT_STATUS=$?
fi

coverage combine -q
coverage xml -i

# Copy coverage report for Codecov artifact upload
mkdir -p artifacts-to-be-uploaded
cp coverage.xml artifacts-to-be-uploaded/ || true

# ==================================================================================== #
# ================================ Post-proc ========================================= #

bash ${this_dir}/post_process.sh

# Exit with failure if any tests failed
exit $EXIT_STATUS
