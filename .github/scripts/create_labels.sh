#!/bin/bash
# Script to create CI granularity labels in the GitHub repository
# Usage: .github/scripts/create_labels.sh [owner/repo]
# Example: .github/scripts/create_labels.sh pytorch/rl
#
# This script focuses on the NEW granular CI labels.
# Existing labels (bug, enhancement, etc.) are not modified.

set -euo pipefail

REPO="${1:-pytorch/rl}"

echo "Creating/updating CI granularity labels for repository: $REPO"
echo "=============================================================="

# =============================================================================
# Environment labels (green) - NEW granular labels
# =============================================================================
ENV_COLOR="0E8A16"

echo ""
echo "Creating Environments labels..."
# Parent label (may already exist)
gh label create "Environments" --repo "$REPO" --description "Triggers all environment CI tests" --color "$ENV_COLOR" --force

# New granular labels
gh label create "Environments/brax" --repo "$REPO" --description "Triggers brax environment tests only" --color "$ENV_COLOR" --force
gh label create "Environments/mujoco_playground" --repo "$REPO" --description "Triggers mujoco_playground environment tests only" --color "$ENV_COLOR" --force
gh label create "Environments/chess" --repo "$REPO" --description "Triggers chess environment tests only" --color "$ENV_COLOR" --force
gh label create "Environments/envpool" --repo "$REPO" --description "Triggers envpool environment tests only" --color "$ENV_COLOR" --force
gh label create "Environments/isaaclab" --repo "$REPO" --description "Triggers Isaac Lab environment tests only" --color "$ENV_COLOR" --force
gh label create "Environments/jumanji" --repo "$REPO" --description "Triggers jumanji environment tests only" --color "$ENV_COLOR" --force
gh label create "Environments/meltingpot" --repo "$REPO" --description "Triggers meltingpot environment tests only" --color "$ENV_COLOR" --force
gh label create "Environments/open_spiel" --repo "$REPO" --description "Triggers open_spiel environment tests only" --color "$ENV_COLOR" --force
gh label create "Environments/pettingzoo" --repo "$REPO" --description "Triggers pettingzoo environment tests only" --color "$ENV_COLOR" --force
gh label create "Environments/procgen" --repo "$REPO" --description "Triggers procgen environment tests only" --color "$ENV_COLOR" --force
gh label create "Environments/robohive" --repo "$REPO" --description "Triggers robohive environment tests only" --color "$ENV_COLOR" --force
gh label create "Environments/smacv2" --repo "$REPO" --description "Triggers smacv2 environment tests only" --color "$ENV_COLOR" --force
gh label create "Environments/unity_mlagents" --repo "$REPO" --description "Triggers Unity ML-Agents environment tests only" --color "$ENV_COLOR" --force
gh label create "Environments/vmas" --repo "$REPO" --description "Triggers vmas environment tests only" --color "$ENV_COLOR" --force

# =============================================================================
# Data labels (blue) - NEW granular labels
# =============================================================================
DATA_COLOR="1D76DB"

echo ""
echo "Creating Data labels..."
# Parent label (may already exist)
gh label create "Data" --repo "$REPO" --description "Triggers all data CI tests" --color "$DATA_COLOR" --force

# New granular labels
gh label create "Data/gendgrl" --repo "$REPO" --description "Triggers gen-dgrl data tests only" --color "$DATA_COLOR" --force
gh label create "Data/minari" --repo "$REPO" --description "Triggers minari data tests only" --color "$DATA_COLOR" --force
gh label create "Data/openx" --repo "$REPO" --description "Triggers openx data tests only" --color "$DATA_COLOR" --force
gh label create "Data/roboset" --repo "$REPO" --description "Triggers roboset data tests only" --color "$DATA_COLOR" --force
gh label create "Data/vd4rl" --repo "$REPO" --description "Triggers vd4rl data tests only" --color "$DATA_COLOR" --force

# =============================================================================
# ciflow/* labels (grey) - opt-in escape hatches for selective PR CI
# See .github/CI.md for the full list of tracks.
# =============================================================================
CIFLOW_COLOR="BFDADC"

echo ""
echo "Creating ciflow/* labels..."
gh label create "ciflow/full"       --repo "$REPO" --description "Run the full Linux test matrix on the PR (all tracks, all Python versions, all shards)" --color "$CIFLOW_COLOR" --force
gh label create "ciflow/cpu-matrix" --repo "$REPO" --description "Run all Python versions on the CPU test job (3.10..3.14)"                               --color "$CIFLOW_COLOR" --force
gh label create "ciflow/gpu"        --repo "$REPO" --description "Run all three GPU test shards on the PR"                                               --color "$CIFLOW_COLOR" --force
gh label create "ciflow/stable"     --repo "$REPO" --description "Run tests-stable-gpu + tests-stable-gpu-distributed on the PR"                          --color "$CIFLOW_COLOR" --force
gh label create "ciflow/olddeps"    --repo "$REPO" --description "Run tests-olddeps on the PR"                                                            --color "$CIFLOW_COLOR" --force
gh label create "ciflow/optdeps"    --repo "$REPO" --description "Run tests-optdeps on the PR"                                                            --color "$CIFLOW_COLOR" --force
gh label create "ciflow/distributed" --repo "$REPO" --description "Run tests-gpu-distributed on the PR"                                                   --color "$CIFLOW_COLOR" --force
gh label create "ciflow/sota"       --repo "$REPO" --description "Run test-linux-sota on the PR"                                                          --color "$CIFLOW_COLOR" --force
gh label create "ciflow/tutorials"  --repo "$REPO" --description "Run test-linux-tutorials on the PR"                                                     --color "$CIFLOW_COLOR" --force
gh label create "ciflow/windows"    --repo "$REPO" --description "Run test-windows-optdepts on the PR"                                                    --color "$CIFLOW_COLOR" --force

echo ""
echo "=============================================================="
echo "Done! All CI granularity labels created/updated."
echo ""
echo "Usage:"
echo "  - 'Environments' → triggers ALL environment tests"
echo "  - 'Environments/brax' → triggers only brax tests"
echo "  - 'Data' → triggers ALL data tests"
echo "  - 'Data/minari' → triggers only minari tests"
echo "  - 'ciflow/full' → run the full Linux matrix on the PR"
echo "  - 'ciflow/gpu' → run all 3 GPU shards on the PR"
echo "  - see .github/CI.md for the complete ciflow/* list"
