#!/bin/bash
# Build TorchRL documentation locally using uv with a temporary virtual environment.
#
# Usage:
#   ./build_local.sh          # Build docs (runs tutorials)
#   ./build_local.sh --no-run # Build docs without running tutorials (faster)
#
# The script creates a temporary virtual environment, installs dependencies,
# builds the documentation, and cleans up on failure.

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${ROOT_DIR}/.doc-venv"
BUILD_DIR="${SCRIPT_DIR}/_local_build"
PYTHON_VERSION="3.12"

# Parse arguments
RUN_TUTORIALS=true
for arg in "$@"; do
    case $arg in
        --no-run)
            RUN_TUTORIALS=false
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--no-run]"
            echo ""
            echo "Options:"
            echo "  --no-run    Skip running tutorials (faster build)"
            echo "  --help, -h  Show this help message"
            exit 0
            ;;
    esac
done

# Cleanup function - always runs on exit
cleanup() {
    local exit_code=$?
    
    # Restore conf.py if we modified it (do this regardless of success/failure)
    if [ -f "$SCRIPT_DIR/source/conf.py.bak" ]; then
        mv "$SCRIPT_DIR/source/conf.py.bak" "$SCRIPT_DIR/source/conf.py"
        echo "Restored conf.py from backup"
    fi
    
    if [ $exit_code -ne 0 ]; then
        echo ""
        echo "============================================"
        echo "Build failed with exit code $exit_code"
        echo "Cleaning up virtual environment..."
        echo "============================================"
        rm -rf "$VENV_DIR"
        echo "Virtual environment removed: $VENV_DIR"
    fi
    exit $exit_code
}
trap cleanup EXIT INT TERM

echo "============================================"
echo "TorchRL Documentation Build Script"
echo "============================================"
echo ""
echo "Root directory: $ROOT_DIR"
echo "Virtual env: $VENV_DIR"
echo "Build output: $BUILD_DIR"
echo "Run tutorials: $RUN_TUTORIALS"
echo ""

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' is not installed."
    echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Remove existing venv if it exists
if [ -d "$VENV_DIR" ]; then
    echo "Removing existing virtual environment..."
    rm -rf "$VENV_DIR"
fi

# Create virtual environment
echo "Creating virtual environment with Python $PYTHON_VERSION..."
uv venv "$VENV_DIR" --python "$PYTHON_VERSION"

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip and install build tools
echo "Installing build tools..."
uv pip install --upgrade pip setuptools ninja packaging "pybind11[global]" cmake

# Install PyTorch (nightly for latest features)
echo "Installing PyTorch..."
uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu

# Install tensordict from git
echo "Installing tensordict..."
uv pip install git+https://github.com/pytorch/tensordict.git

# Install torchrl in editable mode
echo "Installing torchrl..."
cd "$ROOT_DIR"
uv pip install -e . --no-build-isolation

# Install documentation requirements
echo "Installing documentation requirements..."
# First install pytorch_sphinx_theme separately (editable git not supported by uv -r)
uv pip install git+https://github.com/pytorch/pytorch_sphinx_theme.git
# Install remaining requirements (skip the editable line)
grep -v "pytorch_sphinx_theme" "$SCRIPT_DIR/requirements.txt" | uv pip install -r -

# Verify installation
echo "Verifying torchrl installation..."
python -c "import torchrl; print(f'TorchRL version: {torchrl.__version__}')"

# Set up environment for building
export MAX_IDLE_COUNT=180
export BATCHED_PIPE_TIMEOUT=180
export TORCHRL_CONSOLE_STREAM=stdout

# Clear old MuJoCo environment variables that might interfere with MuJoCo 3.x
unset MUJOCO_PATH
unset MUJOCO_PY_MUJOCO_PATH
unset LD_LIBRARY_PATH  # Reset to avoid old MuJoCo libs

# Set plot_gallery based on mode
# Create backup of conf.py for cleanup to restore
cp "$SCRIPT_DIR/source/conf.py" "$SCRIPT_DIR/source/conf.py.bak"

if [ "$RUN_TUTORIALS" = true ]; then
    echo ""
    echo "Note: Tutorials WILL be executed"
    # Enable plot_gallery (replace both True and "False" variants)
    sed -i '' 's/"plot_gallery": "False"/"plot_gallery": True/' "$SCRIPT_DIR/source/conf.py"
    sed -i '' 's/"plot_gallery": False/"plot_gallery": True/' "$SCRIPT_DIR/source/conf.py"
else
    echo ""
    echo "Note: Tutorials will NOT be executed (--no-run mode)"
    # Disable plot_gallery
    sed -i '' 's/"plot_gallery": True/"plot_gallery": "False"/' "$SCRIPT_DIR/source/conf.py"
fi

# Build documentation
echo ""
echo "============================================"
echo "Building documentation..."
echo "============================================"
cd "$SCRIPT_DIR"
sphinx-build ./source "$BUILD_DIR" -v -j auto

echo ""
echo "============================================"
echo "Documentation built successfully!"
echo "============================================"
echo ""
echo "Output: $BUILD_DIR"
echo ""
echo "To view the documentation, run:"
echo "  python -m http.server 8000 -d $BUILD_DIR"
echo "  # Then open http://localhost:8000 in your browser"
echo ""
echo "To clean up the virtual environment, run:"
echo "  rm -rf $VENV_DIR"
