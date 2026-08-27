#!/usr/bin/env bash
# Assert that the installed PyTorch and TensorDict builds match the CI mode.
#
# Usage:
#   bash assert_torch_tensordict_versions.sh nightly [auto|main|nightly|stable]
#   bash assert_torch_tensordict_versions.sh stable  [auto|main|nightly|stable]
#
# By default (auto), nightly PyTorch expects TensorDict from main/git and stable
# PyTorch expects a stable TensorDict release. Some jobs intentionally pin stable
# PyTorch while testing TensorDict main; pass "main" explicitly for those jobs.
# Nightly wheel jobs install the dated TensorDict nightly package; pass "nightly"
# explicitly for those jobs.
set -euo pipefail

expected_torch="${1:?Usage: assert_torch_tensordict_versions.sh nightly|stable [auto|main|nightly|stable]}"
expected_tensordict="${2:-auto}"

PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"

"$PYTHON" - "$expected_torch" "$expected_tensordict" <<'PY'
from __future__ import annotations

import importlib.metadata
import re
import sys

import tensordict
import torch

expected_torch = sys.argv[1]
expected_tensordict = sys.argv[2]

if expected_torch not in {"nightly", "stable"}:
    raise RuntimeError(
        "Expected PyTorch mode must be 'nightly' or 'stable', "
        f"got {expected_torch!r}."
    )

if expected_tensordict == "auto":
    expected_tensordict = "main" if expected_torch == "nightly" else "stable"
elif expected_tensordict not in {"main", "nightly", "stable"}:
    raise RuntimeError(
        "Expected TensorDict source must be 'auto', 'main', 'nightly', or "
        "'stable', "
        f"got {expected_tensordict!r}."
    )

torch_version = torch.__version__
torch_is_nightly = "dev" in torch_version
expected_torch_is_nightly = expected_torch == "nightly"
if torch_is_nightly != expected_torch_is_nightly:
    raise RuntimeError(
        f"PyTorch version mismatch: expected {expected_torch} but got "
        f"torch {torch_version}."
    )

try:
    tensordict_version = tensordict.__version__
except AttributeError:
    tensordict_version = importlib.metadata.version("tensordict")

try:
    tensordict_dist = importlib.metadata.distribution("tensordict")
    tensordict_direct_url = tensordict_dist.read_text("direct_url.json") or ""
except importlib.metadata.PackageNotFoundError:
    tensordict_direct_url = ""

tensordict_from_main = bool(
    re.search(r"(?:\.dev|\+g|dev)", tensordict_version)
    or "github.com/pytorch/tensordict" in tensordict_direct_url
)
tensordict_from_nightly = bool(re.match(r"^\d{4}\.\d{2}\.\d{2}", tensordict_version))

if expected_tensordict == "main" and not tensordict_from_main:
    raise RuntimeError(
        "TensorDict source mismatch: expected TensorDict from main/git but got "
        f"tensordict {tensordict_version}."
    )

if expected_tensordict == "nightly" and not tensordict_from_nightly:
    raise RuntimeError(
        "TensorDict source mismatch: expected a TensorDict nightly release but "
        f"got tensordict {tensordict_version}."
    )

if expected_tensordict == "stable" and (
    tensordict_from_main or tensordict_from_nightly
):
    raise RuntimeError(
        "TensorDict source mismatch: expected a stable TensorDict release but got "
        f"tensordict {tensordict_version}."
    )

print(f"PyTorch version check OK: torch {torch_version} (expected {expected_torch})")
print(
    "TensorDict source check OK: "
    f"tensordict {tensordict_version} (expected {expected_tensordict})"
)
PY
