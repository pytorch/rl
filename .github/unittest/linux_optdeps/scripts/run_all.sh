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
  # `dist-upgrade` of the throwaway container adds time to every job and
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
pip3 install cloudpickle packaging importlib_metadata numpy orjson "pyvers>=0.2.3,<0.3.0"
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

# PR smoke mode (tests-optdeps-smoke): prove the optional-dependency
# environment builds and imports cleanly, then stop before the full suite.
# The full suite runs on main/nightly, or on PRs with the ci/optdeps label.
if [ "${TORCHRL_OPTDEPS_SMOKE:-0}" = "1" ]; then
  python -m pytest test/smoke_test.py -v --durations 20
  echo "Optdeps smoke mode: imports OK, skipping the full suite."
  exit 0
fi

# Track test failures but keep going, so coverage is still combined and
# uploaded; the script exits with this status at the end. (Previously the
# script aborted on the pytest failure via set -e, before coverage upload.)
EXIT_STATUS=0

# Hang containment (an uninterruptible C-level teardown once consumed the
# full two-hour job budget while producing no output). A fixed wall-clock
# guard cannot work here: healthy runs have been measured anywhere between
# 76 and 105+ minutes of pytest, so any budget tight enough to save the job
# also kills slow-but-progressing runs. Neither can pytest-timeout's thread
# method: its killer thread needs the GIL, which the native hang holds, and
# it turns any legitimately slow test into a whole-run abort. The reliable
# signal is inactivity -- the suite prints a line every few seconds under
# -vv while the observed hang was a full hour of silence -- so a watchdog
# kills the run only after 20 minutes without output.
pytest_log="${root_dir}/optdeps-pytest.log"
stall_flag="${root_dir}/optdeps-stalled"
rm -f "${stall_flag}"
: > "${pytest_log}"
# Line-buffer python output through the tee pipe so the log's mtime tracks
# real pytest activity (and CI log streaming stays live).
export PYTHONUNBUFFERED=1

# setsid gives pytest its own process group so the watchdog can kill the
# whole tree without touching this script.
setsid bash -c '
python .github/unittest/helpers/coverage_run_parallel.py -m pytest test \
  --instafail --durations 200 -vv --capture no --ignore test/test_rlhf.py \
  --ignore test/test_distributed.py \
  --ignore test/llm \
  --timeout=120 --mp_fork_if_no_cuda
' > >(tee "${pytest_log}") 2>&1 &
pytest_pid=$!

(
  set +x
  stall_limit=$((20 * 60))
  while kill -0 "${pytest_pid}" 2>/dev/null; do
    sleep 60
    last_output=$(stat -c %Y "${pytest_log}" 2>/dev/null || echo 0)
    if [ $(( $(date +%s) - last_output )) -ge "${stall_limit}" ]; then
      touch "${stall_flag}"
      echo "No pytest output for ${stall_limit}s; killing the hung run"
      kill -TERM -- "-${pytest_pid}" 2>/dev/null
      sleep 60
      kill -KILL -- "-${pytest_pid}" 2>/dev/null
      break
    fi
  done
) &
watchdog_pid=$!

wait "${pytest_pid}" || EXIT_STATUS=$?
kill "${watchdog_pid}" 2>/dev/null || true
if [ -f "${stall_flag}" ]; then
  echo "pytest was killed by the 20-minute inactivity watchdog"
elif [ $EXIT_STATUS -ne 0 ]; then
  echo "Some tests failed with exit status $EXIT_STATUS"
fi

# Tolerate combine failures: a run killed by the watchdog may leave no
# usable coverage data, and the remaining artifacts should still upload.
coverage combine -q || echo "coverage combine failed (expected after a watchdog kill)"
coverage xml -i || true

# Copy coverage report for Codecov artifact upload
mkdir -p artifacts-to-be-uploaded
cp coverage.xml artifacts-to-be-uploaded/ || true

# ==================================================================================== #
# ================================ Post-proc ========================================= #

bash ${this_dir}/post_process.sh

# Exit with failure if any tests failed
exit $EXIT_STATUS
