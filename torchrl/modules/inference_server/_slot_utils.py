# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from collections.abc import MutableSequence
from ctypes import Array, c_byte

import torch
from tensordict.base import _is_leaf_nontensor, TensorDictBase


def _make_slot_bank(
    spec: TensorDictBase, num_slots: int, transport_name: str, argname: str
) -> TensorDictBase:
    """Validate a CPU tensor spec and allocate a shared request or response bank."""
    # _is_leaf_nontensor surfaces NonTensorData leaves (excluded by the
    # default leaf iterator) so they are rejected instead of ignored.
    leaves = list(
        spec.items(include_nested=True, leaves_only=True, is_leaf=_is_leaf_nontensor)
    )
    if not leaves:
        raise ValueError(f"{argname} must contain at least one tensor leaf.")
    for key, value in leaves:
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"{transport_name} specs only support tensor leaves; "
                f"{argname} has a {type(value).__name__} at key {key!r}. "
                "Encode small metadata as tensors or use MPTransport."
            )
        if value.device.type != "cpu":
            raise ValueError(
                f"{transport_name} slots live in CPU shared memory; "
                f"{argname} has a {value.device} tensor at key {key!r}."
            )
    return spec.unsqueeze(0).expand(num_slots, *spec.batch_size).clone().share_memory_()


def _take_ready_slots(
    ready: MutableSequence[bool] | Array[c_byte], start: int, max_items: int
) -> list[int]:
    """Claim ready slots in circular order; the caller supplies synchronization."""
    slots = []
    if max_items <= 0:
        return slots
    for offset in range(len(ready)):
        slot = (start + offset) % len(ready)
        if ready[slot]:
            ready[slot] = False
            slots.append(slot)
            if len(slots) >= max_items:
                break
    return slots
