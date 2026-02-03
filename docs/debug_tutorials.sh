#!/bin/bash
# Debug script to run tutorials one at a time with plain python calls.
# This helps isolate errors that occur in the sphinx-gallery environment.
#
# Usage:
#   ./debug_tutorials.sh                    # Run all tutorials
#   ./debug_tutorials.sh tutorial_name.py   # Run a specific tutorial
#   ./debug_tutorials.sh --list             # List available tutorials
#   ./debug_tutorials.sh --setup-only       # Only set up the environment
#
# The script creates the same virtual environment as build_local.sh.

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${ROOT_DIR}/.doc-venv"
TUTORIALS_DIR="${ROOT_DIR}/tutorials/sphinx-tutorials"
PYTHON_VERSION="3.12"

# Parse arguments
SPECIFIC_TUTORIAL=""
LIST_ONLY=false
SETUP_ONLY=false

for arg in "$@"; do
    case $arg in
        --list)
            LIST_ONLY=true
            shift
            ;;
        --setup-only)
            SETUP_ONLY=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS] [tutorial_name.py]"
            echo ""
            echo "Options:"
            echo "  --list        List available tutorials"
            echo "  --setup-only  Only set up the environment, don't run tutorials"
            echo "  --help, -h    Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                      # Run all tutorials"
            echo "  $0 coding_ppo.py        # Run only coding_ppo.py"
            echo "  $0 --list               # List tutorials"
            exit 0
            ;;
        *.py)
            SPECIFIC_TUTORIAL="$arg"
            shift
            ;;
    esac
done

# List tutorials and exit if requested
if [ "$LIST_ONLY" = true ]; then
    echo "Available tutorials in $TUTORIALS_DIR:"
    echo ""
    ls -1 "$TUTORIALS_DIR"/*.py 2>/dev/null | xargs -n1 basename | sort
    exit 0
fi

echo "============================================"
echo "TorchRL Tutorial Debug Script"
echo "============================================"
echo ""
echo "Root directory: $ROOT_DIR"
echo "Virtual env: $VENV_DIR"
echo "Tutorials dir: $TUTORIALS_DIR"
echo ""

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' is not installed."
    echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Setup virtual environment if it doesn't exist
setup_venv() {
    echo "============================================"
    echo "Setting up virtual environment..."
    echo "============================================"
    
    # Remove existing venv if it exists
    if [ -d "$VENV_DIR" ]; then
        echo "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    fi
    
    # Create virtual environment
    echo "Creating virtual environment with Python $PYTHON_VERSION..."
    uv venv "$VENV_DIR" --python "$PYTHON_VERSION"
    
    # Activate virtual environment
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
    uv pip install git+https://github.com/pytorch/pytorch_sphinx_theme.git
    grep -v "pytorch_sphinx_theme" "$SCRIPT_DIR/requirements.txt" | uv pip install -r -
    
    # Clear MuJoCo environment variables to avoid conflicts
    unset MUJOCO_PATH
    unset MUJOCO_PY_MUJOCO_PATH
    unset LD_LIBRARY_PATH
    
    echo ""
    echo "Virtual environment setup complete!"
}

# Check if venv exists and is valid
if [ ! -d "$VENV_DIR" ] || [ ! -f "$VENV_DIR/bin/activate" ]; then
    setup_venv
else
    echo "Using existing virtual environment: $VENV_DIR"
    echo "(Use 'rm -rf $VENV_DIR' to force recreation)"
    echo ""
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Clear MuJoCo environment variables
unset MUJOCO_PATH
unset MUJOCO_PY_MUJOCO_PATH
unset LD_LIBRARY_PATH

# Verify installation
echo "Verifying torchrl installation..."
python -c "import torchrl; print(f'TorchRL version: {torchrl.__version__}')"
echo ""

if [ "$SETUP_ONLY" = true ]; then
    echo "Setup complete. Virtual environment is ready at: $VENV_DIR"
    echo "To activate manually: source $VENV_DIR/bin/activate"
    exit 0
fi

# Tutorials that can't be run directly or have known issues
SKIP_TUTORIALS=(
    "llm_browser.py"    # Requires Playwright browser and async operations
    "torchrl_demo.py"   # Uses share_memory_() demos that can fail if mp subsystem is corrupted
)

# Get list of tutorials to run
if [ -n "$SPECIFIC_TUTORIAL" ]; then
    if [ -f "$TUTORIALS_DIR/$SPECIFIC_TUTORIAL" ]; then
        TUTORIALS=("$TUTORIALS_DIR/$SPECIFIC_TUTORIAL")
    else
        echo "Error: Tutorial not found: $TUTORIALS_DIR/$SPECIFIC_TUTORIAL"
        exit 1
    fi
else
    TUTORIALS=($(ls "$TUTORIALS_DIR"/*.py 2>/dev/null | sort))
fi

# Filter out skipped tutorials (unless specifically requested)
if [ -z "$SPECIFIC_TUTORIAL" ]; then
    FILTERED_TUTORIALS=()
    for tutorial in "${TUTORIALS[@]}"; do
        tutorial_name=$(basename "$tutorial")
        skip=false
        for skip_name in "${SKIP_TUTORIALS[@]}"; do
            if [ "$tutorial_name" = "$skip_name" ]; then
                skip=true
                echo "Skipping $tutorial_name (multiprocessing/external dependencies)"
                break
            fi
        done
        if [ "$skip" = false ]; then
            FILTERED_TUTORIALS+=("$tutorial")
        fi
    done
    TUTORIALS=("${FILTERED_TUTORIALS[@]}")
    echo ""
fi

echo "============================================"
echo "Running ${#TUTORIALS[@]} tutorial(s)..."
echo "============================================"
echo ""

# Track results
PASSED=()
FAILED=()

# Run each tutorial
for tutorial in "${TUTORIALS[@]}"; do
    tutorial_name=$(basename "$tutorial")
    echo "--------------------------------------------"
    echo "Running: $tutorial_name"
    echo "--------------------------------------------"
    
    # Create a temporary directory for the tutorial to use
    TUTORIAL_TMPDIR=$(mktemp -d)
    
    # Run the tutorial with a timeout
    set +e  # Don't exit on error
    cd "$TUTORIAL_TMPDIR"
    
    # Set environment for tutorial
    export MPLBACKEND=Agg  # Use non-interactive matplotlib backend
    
    # Run with timeout (5 minutes per tutorial)
    timeout 300 python "$tutorial" 2>&1
    exit_code=$?
    
    set -e  # Re-enable exit on error
    
    # Clean up temp directory
    rm -rf "$TUTORIAL_TMPDIR"
    
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "✓ PASSED: $tutorial_name"
        PASSED+=("$tutorial_name")
    elif [ $exit_code -eq 124 ]; then
        echo ""
        echo "✗ TIMEOUT: $tutorial_name (exceeded 5 minutes)"
        FAILED+=("$tutorial_name (timeout)")
    else
        echo ""
        echo "✗ FAILED: $tutorial_name (exit code: $exit_code)"
        FAILED+=("$tutorial_name")
    fi
    echo ""
done

# Print summary
echo "============================================"
echo "Summary"
echo "============================================"
echo ""
echo "Passed: ${#PASSED[@]}"
for t in "${PASSED[@]}"; do
    echo "  ✓ $t"
done
echo ""
echo "Failed: ${#FAILED[@]}"
for t in "${FAILED[@]}"; do
    echo "  ✗ $t"
done
echo ""

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Some tutorials failed. Run individual tutorials to debug:"
    echo "  $0 <tutorial_name.py>"
    exit 1
else
    echo "All tutorials passed!"
    exit 0
fi
