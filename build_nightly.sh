#!/bin/bash
set -e

# Check if we're in a nightly build
if [ "$TORCHRL_NIGHTLY" != "1" ]; then
    echo "Not a nightly build, exiting"
    exit 0
fi

echo "Starting nightly build process..."

# Create backups of original files
cp pyproject.toml pyproject.toml.backup
cp version.txt version.txt.backup

# Function to restore original files
restore_files() {
    echo "Restoring original files..."
    mv pyproject.toml.backup pyproject.toml
    mv version.txt.backup version.txt
}

# Set up trap to restore files on exit (success or failure)
trap restore_files EXIT

# Modify pyproject.toml for nightly build
echo "Modifying pyproject.toml for nightly build..."
sed -i.bak 's/name = "torchrl"/name = "torchrl-nightly"/' pyproject.toml

# Replace tensordict dependency with tensordict-nightly
echo "Replacing tensordict with tensordict-nightly..."
sed -i.bak 's/"tensordict[^"]*"/"tensordict-nightly"/g' pyproject.toml

# Clean up sed backup files
rm -f pyproject.toml.bak

# Set nightly version (YYYY.MM.DD format)
echo "Setting nightly version..."
echo "$(date +%Y.%m.%d)" > version.txt

# Build the package
echo "Building nightly package..."
echo "Python version: $(python --version)"
echo "Platform: $(python -c 'import sys; print(sys.platform)')"

# Install build dependencies
echo "Installing build dependencies..."
# Use setuptools 65.3.0 for Python <= 3.10 (fixes compatibility issues)
# Use latest setuptools for Python 3.11+
# 
# Rationale:
# - Python 3.9/3.10: setuptools 65.3.0 is more stable and fixes version detection issues
#   that cause "0.0.0+unknown" version strings in nightly builds
# - Python 3.11+: Newer setuptools versions are required for proper functionality
#   with the latest Python features and build system changes
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if python -c "import sys; exit(0 if sys.version_info < (3, 11) else 1)"; then
    echo "Using setuptools 65.3.0 for Python $PYTHON_VERSION (compatibility mode)"
    python -m pip install wheel setuptools==65.3.0 "pybind11[global]"
else
    echo "Using latest setuptools for Python $PYTHON_VERSION (modern mode)"
    python -m pip install wheel setuptools "pybind11[global]"
fi

python setup.py bdist_wheel

echo "Nightly build completed successfully!" 
