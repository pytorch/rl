# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Constants and helper classes for collectors."""
from __future__ import annotations

import os
import sys
from multiprocessing.managers import SyncManager

import torch
from torch import multiprocessing as mp

from torchrl.envs.utils import ExplorationType

try:
    from torch.compiler import cudagraph_mark_step_begin
except ImportError:

    def cudagraph_mark_step_begin():
        """Placeholder for missing cudagraph_mark_step_begin method."""
        raise NotImplementedError("cudagraph_mark_step_begin not implemented.")


__all__ = [
    "_TIMEOUT",
    "INSTANTIATE_TIMEOUT",
    "_MIN_TIMEOUT",
    "_MAX_IDLE_COUNT",
    "WEIGHT_SYNC_TIMEOUT",
    "DEFAULT_EXPLORATION_TYPE",
    "_is_osx",
    "_Interruptor",
    "_InterruptorManager",
    "cudagraph_mark_step_begin",
]

_TIMEOUT = 1.0
INSTANTIATE_TIMEOUT = 20
_MIN_TIMEOUT = 1e-3  # should be several orders of magnitude inferior wrt time spent collecting a trajectory
# Timeout for weight synchronization during collector init.
# Increase this when using many collectors across different CUDA devices.
WEIGHT_SYNC_TIMEOUT = float(os.environ.get("TORCHRL_WEIGHT_SYNC_TIMEOUT", 120.0))
# MAX_IDLE_COUNT is the maximum number of times a Dataloader worker can timeout with his queue.
_MAX_IDLE_COUNT = int(os.environ.get("MAX_IDLE_COUNT", torch.iinfo(torch.int64).max))

DEFAULT_EXPLORATION_TYPE: ExplorationType = ExplorationType.RANDOM

_is_osx = sys.platform.startswith("darwin")


class _Interruptor:
    """A shared-memory flag for interrupting rollout collection across processes.

    The main process raises the flag with ``start_collection`` and clears it
    with ``stop_collection``; workers poll ``collection_stopped`` once per env
    step and cut their rollout short when it returns ``True``.

    The flag is a single shared-memory byte with exactly one writer (the main
    process), so no lock is needed: one-byte reads cannot be torn and workers
    only ever read. Like any ``multiprocessing.sharedctypes`` value, the object
    must be handed to workers at process-creation time (it is inherited and
    cannot be sent through queues or pipes afterwards).
    """

    # interrupter vs interruptor: google trends seems to indicate that "or" is more
    # widely used than "er" even if my IDE complains about that...
    def __init__(self):
        self._collect = mp.Value("b", True, lock=False)

    def start_collection(self):
        self._collect.value = True

    def stop_collection(self):
        self._collect.value = False

    def collection_stopped(self):
        return not self._collect.value


class _InterruptorManager(SyncManager):
    """A custom SyncManager for managing the collection state of a process.

    This class extends the SyncManager class and allows to share an Interruptor object
    between processes.

    .. note::
        No longer used internally: :class:`_Interruptor` is now backed by shared
        memory and inherited directly by worker processes. Kept for backward
        compatibility.
    """


_InterruptorManager.register("_Interruptor", _Interruptor)
