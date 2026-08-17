# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from torchrl.modules.tensordict_module.exploration import RandomPolicy

from ._async_batched import AsyncBatchedCollector

from ._base import BaseCollector, ProfileConfig
from ._evaluator import Evaluator

from ._multi_async import MultiAsyncCollector
from ._multi_base import MultiCollector
from ._multi_sync import MultiSyncCollector
from ._single import Collector
from ._single_async import AsyncCollector
from .weight_update import (
    MultiProcessedWeightUpdater,
    RayWeightUpdater,
    RemoteModuleWeightUpdater,
    VanillaWeightUpdater,
    WeightUpdaterBase,
)

__all__ = [
    # Shared collector API
    "BaseCollector",
    # Main construction API
    "Collector",
    # Specialized and concrete implementations
    "AsyncCollector",
    "MultiCollector",
    "MultiSyncCollector",
    "AsyncBatchedCollector",
    "Evaluator",
    "MultiAsyncCollector",
    "ProfileConfig",
    # Other exports
    "WeightUpdaterBase",
    "VanillaWeightUpdater",
    "RandomPolicy",
    "RayWeightUpdater",
    "RemoteModuleWeightUpdater",
    "MultiProcessedWeightUpdater",
]
