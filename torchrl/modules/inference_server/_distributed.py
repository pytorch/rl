# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from torchrl._comm.distributed import _DistributedTransport
from torchrl.modules.inference_server._transport import InferenceTransport


class _DistributedInferenceTransport(_DistributedTransport, InferenceTransport):
    """Private inference adapter for the tensor request/reply transport."""


__all__ = ["_DistributedInferenceTransport"]
