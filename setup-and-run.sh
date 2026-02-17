#!/usr/bin/env bash
# =============================================================================
# setup-and-run.sh — Idempotent setup for running Dreamer on steve
#
# Usage:
#   # Setup + Isaac training (default):
#   bash setup-and-run.sh
#
#   # Setup only (validate deps, don't train):
#   bash setup-and-run.sh --build-only
#
#   # DMControl training (original Dreamer, needs MuJoCo):
#   bash setup-and-run.sh --dmcontrol
#
#   # Override Isaac task:
#   bash setup-and-run.sh env.name=Isaac-Ant-v0
#
# Steve workflow:
#   JOBID=$(steve job --partition h200-high --gpus-per-task 1 --ntasks 1 \
#     --time 24:00:00 --job-name "dreamer-isaac" --container-image nvcr.io/nvidia/isaac-lab:2.3.0 --jobid-only)
#   steve cp "$JOBID" ./setup-and-run.sh :/root/setup-and-run.sh
#   steve step "$JOBID" 'bash /root/setup-and-run.sh --build-only'
#   steve step -d "$JOBID" 'bash /root/setup-and-run.sh'
# =============================================================================
set -euo pipefail

# ---- Configuration ---------------------------------------------------------
REPO_URL="https://github.com/pytorch/rl.git"
REPO_DIR="/root/rl"
VENV_DIR="/root/torchrl_venv"
MODE="isaac"      # "isaac" or "dmcontrol"
BUILD_ONLY=false
EXTRA_ARGS=()      # extra Hydra overrides forwarded to the training script

# ---- Parse arguments --------------------------------------------------------
for arg in "$@"; do
    case "$arg" in
        --build-only)   BUILD_ONLY=true ;;
        --dmcontrol)    MODE="dmcontrol" ;;
        --isaac)        MODE="isaac" ;;
        *)              EXTRA_ARGS+=("$arg") ;;
    esac
done

# Avoid "'': unknown terminal type" in headless containers
export TERM="${TERM:-xterm}"

echo "============================================================"
echo " setup-and-run.sh"
echo "   mode=$MODE  build_only=$BUILD_ONLY"
echo "   extra_args=${EXTRA_ARGS[*]:-<none>}"
echo "============================================================"

# ---- 0) Kill zombie Python processes from previous runs ---------------------
echo "* Killing leftover Python processes..."
pkill -9 -f python || true
sleep 1

# ---- 1) System dependencies ------------------------------------------------
echo "* Installing system packages..."
apt-get update -qq
apt-get install -y -qq git wget gcc g++ > /dev/null 2>&1

# For MuJoCo headless rendering (DMControl mode)
if [[ "$MODE" == "dmcontrol" ]]; then
    apt-get install -y -qq \
        libglfw3 libgl1 libosmesa6 libglew-dev libsdl2-dev libsdl2-2.0-0 \
        libglvnd0 libglx0 libegl1 libgles2 xvfb libegl-dev libx11-dev freeglut3-dev \
        > /dev/null 2>&1
    export MUJOCO_GL=egl
fi

# ---- 2) IsaacLab-specific env vars -----------------------------------------
if [[ "$MODE" == "isaac" ]]; then
    export OMNI_KIT_ACCEPT_EULA=yes
    export PYTHONNOUSERSITE=1
    export TD_GET_DEFAULTS_TO_NONE=1
fi

# ---- 3) Clone / update the repo --------------------------------------------
git config --global --add safe.directory '*'

BRANCH="${BRANCH:-dreamer-isaac}"

if [[ ! -d "$REPO_DIR" ]]; then
    echo "* Cloning repo (branch=$BRANCH)..."
    git clone -b "$BRANCH" "$REPO_URL" "$REPO_DIR"
else
    echo "* Repo exists, pulling latest (branch=$BRANCH)..."
    cd "$REPO_DIR"
    git fetch --all
    git checkout "$BRANCH" 2>/dev/null || true
    git reset --hard "origin/$BRANCH"
fi
cd "$REPO_DIR"

# ---- 4) Python environment -------------------------------------------------
if [[ "$MODE" == "isaac" ]]; then
    # IsaacLab Docker image has its Python at /workspace/isaaclab
    ISAACLAB_DIR="/workspace/isaaclab"
    ISAACLAB_SH="${ISAACLAB_DIR}/isaaclab.sh"

    if [[ ! -f "$ISAACLAB_SH" ]]; then
        echo "ERROR: IsaacLab not found at $ISAACLAB_SH"
        echo "  Are you running inside the nvcr.io/nvidia/isaac-lab:2.3.0 image?"
        exit 1
    fi

    PIP="${ISAACLAB_SH} -p -m pip"
    PYTHON="${ISAACLAB_SH} -p"

    echo "* Using IsaacLab Python environment"
    $PYTHON -c "import isaaclab; print(f'IsaacLab version: OK')"

    # Install tensordict (no-deps to avoid conflicts with Isaac's torch)
    echo "* Installing tensordict..."
    $PIP install "pybind11[global]" --disable-pip-version-check -q
    $PIP install git+https://github.com/pytorch/tensordict.git \
        --no-deps --disable-pip-version-check -q
    $PIP install cloudpickle orjson pyvers --no-deps --disable-pip-version-check -q

    # Install torchrl from local checkout
    echo "* Installing torchrl..."
    $PIP install -e "${REPO_DIR}" --no-build-isolation --no-deps \
        --disable-pip-version-check -q

    # Install runtime deps
    echo "* Installing runtime deps..."
    $PIP install wandb hydra-core omegaconf tqdm --disable-pip-version-check -q
    # ray + moviepy needed for async eval with video rendering
    $PIP install "ray[default]" moviepy --disable-pip-version-check -q

else
    # DMControl mode: create a venv with uv (more reliable than python -m venv)
    if ! command -v uv &>/dev/null; then
        echo "* Installing uv..."
        pip install uv -q
    fi

    if [[ ! -d "$VENV_DIR" ]]; then
        echo "* Creating venv at $VENV_DIR..."
        uv venv "$VENV_DIR" --python 3.11
    fi
    source "$VENV_DIR/bin/activate"

    PIP="pip"
    PYTHON="python"

    echo "* Installing tensordict + torchrl..."
    pip install git+https://github.com/pytorch/tensordict.git -q
    pip install -e "${REPO_DIR}" --no-build-isolation -q
    pip install wandb hydra-core omegaconf tqdm -q

    # MuJoCo / DMControl deps
    pip install mujoco dm_control gymnasium -q
fi

# ---- 5) Verify installation ------------------------------------------------
echo "* Verification:"
$PYTHON -c "
import torch
print(f'  torch={torch.__version__}, cuda={torch.cuda.is_available()}, '
      f'devices={torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    # Quick CUDA sanity check
    x = torch.randn(2, 3, 4, 4, device='cuda')
    y = torch.nn.functional.conv2d(x, torch.randn(3, 3, 3, 3, device='cuda'), padding=1)
    print(f'  CUDA conv OK: {y.shape}')
"

$PYTHON -c "import tensordict; print(f'  tensordict OK')"
$PYTHON -c "import torchrl; print(f'  torchrl OK')"

if [[ "$MODE" == "isaac" ]]; then
    $PYTHON -c "
from isaaclab.app import AppLauncher
import argparse
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args(['--headless'])
AppLauncher(args)
import torch
print(f'  IsaacLab AppLauncher OK')
import gymnasium as gym
import isaaclab_tasks
print(f'  isaaclab_tasks registered')
"
fi

if [[ "$BUILD_ONLY" == true ]]; then
    echo "============================================================"
    echo " Build-only mode: all deps verified. Exiting."
    echo "============================================================"
    exit 0
fi

# ---- 6) Run training -------------------------------------------------------
echo "============================================================"
echo " Starting training (mode=$MODE)"
echo "============================================================"

cd "$REPO_DIR"

if [[ "$MODE" == "isaac" ]]; then
    # Expose 3 GPUs: GPU 0 = sim, GPU 1 = training, GPU 2 = eval (rendering)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
    $PYTHON "sota-implementations/dreamer/dreamer_isaac.py" "${EXTRA_ARGS[@]}"
else
    export MUJOCO_GL=egl
    $PYTHON "sota-implementations/dreamer/dreamer.py" "${EXTRA_ARGS[@]}"
fi
