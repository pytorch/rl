#!/bin/bash
set -euo pipefail
# Run from the repository root; each seed records its configuration and returns.
for seed in 0 1 2; do
  python sota-implementations/gtrxl/gtrxl_ppo.py seed="$seed" \
    logger.output="outputs/gtrxl/seed-${seed}.json" "$@"
done
