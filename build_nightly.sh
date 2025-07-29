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
python -m build

echo "Nightly build completed successfully!" 