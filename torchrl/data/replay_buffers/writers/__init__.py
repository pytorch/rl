# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from .base import (
    _GENERATION_EAGER_ALLOC_LIMIT as _GENERATION_EAGER_ALLOC_LIMIT,
    _GENERATION_MIN_ALLOC as _GENERATION_MIN_ALLOC,
    _SLOT_GENERATIONS_ATTR as _SLOT_GENERATIONS_ATTR,
    ImmutableDatasetWriter,
    Writer,
)
from .ensemble import WriterEnsemble
from .max_value import TensorDictMaxValueWriter
from .round_robin import RoundRobinWriter, TensorDictRoundRobinWriter

__all__ = [
    "ImmutableDatasetWriter",
    "RoundRobinWriter",
    "TensorDictMaxValueWriter",
    "TensorDictRoundRobinWriter",
    "Writer",
    "WriterEnsemble",
]

for _export in (
    Writer,
    ImmutableDatasetWriter,
    RoundRobinWriter,
    TensorDictRoundRobinWriter,
    TensorDictMaxValueWriter,
    WriterEnsemble,
):
    _export.__module__ = __name__
del _export
