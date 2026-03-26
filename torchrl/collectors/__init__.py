# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from torchrl.modules.tensordict_module.exploration import RandomPolicy

from ._async_batched import AsyncBatchedCollector

from ._base import BaseCollector, DataCollectorBase, ProfileConfig

from ._multi_async import MultiAsyncCollector, MultiaSyncDataCollector
from ._multi_base import MultiCollector, MultiCollector as _MultiDataCollector
from ._multi_sync import MultiSyncCollector, MultiSyncDataCollector
from ._single import Collector, SyncDataCollector
from ._single_async import AsyncCollector, aSyncDataCollector
from .weight_update import (
    MultiProcessedWeightUpdater,
    RayWeightUpdater,
    RemoteModuleWeightUpdater,
    VanillaWeightUpdater,
    WeightUpdaterBase,
)

__all__ = [
    # New canonical names (preferred)
    "BaseCollector",
    "Collector",
    "AsyncCollector",
    "MultiCollector",
    "MultiSyncCollector",
    "AsyncBatchedCollector",
    "MultiAsyncCollector",
    "ProfileConfig",
    # Legacy names (backward-compatible aliases)
    "DataCollectorBase",
    "SyncDataCollector",
    "aSyncDataCollector",
    "_MultiDataCollector",
    "MultiSyncDataCollector",
    "MultiaSyncDataCollector",
    # Other exports
    "WeightUpdaterBase",
    "VanillaWeightUpdater",
    "RandomPolicy",
    "RayWeightUpdater",
    "RemoteModuleWeightUpdater",
    "MultiProcessedWeightUpdater",
]
