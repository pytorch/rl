#!/bin/bash
# Script to create all labels in the GitHub repository
# Usage: .github/scripts/create_labels.sh [owner/repo]
# Example: .github/scripts/create_labels.sh pytorch/rl

set -euo pipefail

REPO="${1:-pytorch/rl}"

echo "Creating labels for repository: $REPO"
echo "================================================"

# =============================================================================
# Issue/PR Type Labels
# =============================================================================
echo "Creating issue/PR type labels..."
gh label create "bug" --repo "$REPO" --description "Something isn't working" --color "d73a4a" --force
gh label create "enhancement" --repo "$REPO" --description "New feature or request" --color "a2eeef" --force
gh label create "documentation" --repo "$REPO" --description "Improvements or additions to documentation" --color "0075ca" --force
gh label create "question" --repo "$REPO" --description "Further information is requested" --color "d876e3" --force
gh label create "good first issue" --repo "$REPO" --description "Good for newcomers" --color "7057ff" --force
gh label create "help wanted" --repo "$REPO" --description "Extra attention is needed" --color "008672" --force
gh label create "duplicate" --repo "$REPO" --description "This issue or pull request already exists" --color "cfd3d7" --force
gh label create "wontfix" --repo "$REPO" --description "This will not be worked on" --color "ffffff" --force
gh label create "invalid" --repo "$REPO" --description "This doesn't seem right" --color "e4e669" --force

# =============================================================================
# Environment labels (green)
# =============================================================================
ENV_COLOR="0E8A16"

echo "Creating Environments labels..."
gh label create "Environments" --repo "$REPO" --description "Triggers all environment CI tests" --color "$ENV_COLOR" --force
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
# Data labels (blue)
# =============================================================================
DATA_COLOR="1D76DB"

echo "Creating Data labels..."
gh label create "Data" --repo "$REPO" --description "Triggers all data CI tests" --color "$DATA_COLOR" --force
gh label create "Data/gendgrl" --repo "$REPO" --description "Triggers gen-dgrl data tests only" --color "$DATA_COLOR" --force
gh label create "Data/minari" --repo "$REPO" --description "Triggers minari data tests only" --color "$DATA_COLOR" --force
gh label create "Data/openx" --repo "$REPO" --description "Triggers openx data tests only" --color "$DATA_COLOR" --force
gh label create "Data/roboset" --repo "$REPO" --description "Triggers roboset data tests only" --color "$DATA_COLOR" --force
gh label create "Data/vd4rl" --repo "$REPO" --description "Triggers vd4rl data tests only" --color "$DATA_COLOR" --force

# =============================================================================
# LLM labels (yellow)
# =============================================================================
LLM_COLOR="FBCA04"

echo "Creating LLM labels..."
gh label create "llm/" --repo "$REPO" --description "Triggers LLM CI tests" --color "$LLM_COLOR" --force

# =============================================================================
# Benchmark labels (purple)
# =============================================================================
BENCH_COLOR="5319E7"

echo "Creating Benchmark labels..."
gh label create "benchmarks/upload" --repo "$REPO" --description "Uploads benchmark results on PR" --color "$BENCH_COLOR" --force

echo ""
echo "================================================"
echo "Done! All labels created successfully."
echo ""
echo "CI Label usage:"
echo "  - 'Environments' → triggers ALL environment tests"
echo "  - 'Environments/brax' → triggers only brax tests"
echo "  - 'Data' → triggers ALL data tests"
echo "  - 'Data/minari' → triggers only minari tests"
echo "  - 'llm/' → triggers LLM tests"
echo "  - 'benchmarks/upload' → uploads benchmark results"
