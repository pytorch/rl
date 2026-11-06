# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Any

import torch
from tensordict import NestedKey, TensorDictBase


def _validate_export_dtypes(
    batch: Any, *, dtypes: dict[NestedKey, torch.dtype]
) -> None:
    # Validate before stacking: torch.stack can silently round mixed dtypes.
    items = batch if isinstance(batch, list) else (batch,)
    for item in items:
        if not isinstance(item, TensorDictBase):
            continue
        for key, value in item.items(include_nested=True, leaves_only=True):
            if isinstance(value, torch.Tensor):
                expected = dtypes.setdefault(key, value.dtype)
                if value.dtype != expected:
                    raise ValueError(
                        f"Tensor at key {key!r} has dtype {value.dtype}, but "
                        f"previous rows have dtype {expected}. Export requires "
                        "a consistent tensor dtype per key."
                    )
