# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .generic import (
    DEFAULT_SLURM_CONF,
    DistributedCollector,
    DistributedDataCollector,
    DistributedWeightUpdater,
)
from .ray import RayCollector
from .rpc import RPCCollector, RPCDataCollector, RPCWeightUpdater
from .sync import DistributedSyncCollector, DistributedSyncDataCollector
from .utils import submitit_delayed_launcher

__all__ = [
    "DEFAULT_SLURM_CONF",
    # New canonical names (preferred)
    "DistributedCollector",
    "DistributedSyncCollector",
    "RPCCollector",
    # Legacy names (backward-compatible aliases)
    "DistributedDataCollector",
    "DistributedSyncDataCollector",
    "RPCDataCollector",
    # Other exports
    "DistributedWeightUpdater",
    "RPCWeightUpdater",
    "RayCollector",
    "submitit_delayed_launcher",
]
