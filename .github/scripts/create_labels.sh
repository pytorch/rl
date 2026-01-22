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

echo ""
echo "=============================================================="
echo "Done! All CI granularity labels created/updated."
echo ""
echo "Usage:"
echo "  - 'Environments' → triggers ALL environment tests"
echo "  - 'Environments/brax' → triggers only brax tests"
echo "  - 'Data' → triggers ALL data tests"
echo "  - 'Data/minari' → triggers only minari tests"
