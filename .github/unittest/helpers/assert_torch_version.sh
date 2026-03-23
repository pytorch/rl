#!/usr/bin/env bash
# Assert that the installed PyTorch version matches the expected type.
#
# Usage:
#   bash assert_torch_version.sh nightly   # fails unless torch version contains "dev"
#   bash assert_torch_version.sh stable    # fails if torch version contains "dev"
set -e

expected="${1:?Usage: assert_torch_version.sh nightly|stable}"

python -c "
import torch
v = torch.__version__
is_nightly = 'dev' in v
expected_nightly = '${expected}' == 'nightly'
if is_nightly != expected_nightly:
    what = 'nightly' if expected_nightly else 'stable'
    raise RuntimeError(
        f'PyTorch version mismatch: expected {what} but got torch {v}'
    )
print(f'PyTorch version check OK: torch {v} (expected ${expected})')
"
