# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Re-exports of collector classes for backward compatibility."""
from __future__ import annotations

# Re-export constants for backward compatibility
from torchrl.collectors._constants import (
    _Interruptor,
    _InterruptorManager,
    _is_osx,
    _MAX_IDLE_COUNT,
    _MIN_TIMEOUT,
    _TIMEOUT,
    cudagraph_mark_step_begin,
    DEFAULT_EXPLORATION_TYPE,
    INSTANTIATE_TIMEOUT,
)

from torchrl.collectors._multi_async import MultiaSyncDataCollector
from torchrl.collectors._multi_base import _MultiDataCollector
from torchrl.collectors._multi_sync import MultiSyncDataCollector
from torchrl.collectors._runner import _main_async_collector
from torchrl.collectors._single import SyncDataCollector
from torchrl.collectors._single_async import aSyncDataCollector
from torchrl.collectors.base import DataCollectorBase

__all__ = [
    "MultiSyncDataCollector",
    "MultiaSyncDataCollector",
    "_MultiDataCollector",
    "SyncDataCollector",
    "_main_async_collector",
    "aSyncDataCollector",
    "DataCollectorBase",
    # Constants
    "_TIMEOUT",
    "INSTANTIATE_TIMEOUT",
    "_MIN_TIMEOUT",
    "_MAX_IDLE_COUNT",
    "DEFAULT_EXPLORATION_TYPE",
    "_is_osx",
    "_Interruptor",
    "_InterruptorManager",
    "cudagraph_mark_step_begin",
]
