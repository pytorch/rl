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
    "DEFAULT_EXPLORATION_TYPE",
    "_is_osx",
    "_Interruptor",
    "_InterruptorManager",
    "cudagraph_mark_step_begin",
]

_TIMEOUT = 1.0
INSTANTIATE_TIMEOUT = 20
_MIN_TIMEOUT = 1e-3  # should be several orders of magnitude inferior wrt time spent collecting a trajectory
# MAX_IDLE_COUNT is the maximum number of times a Dataloader worker can timeout with his queue.
_MAX_IDLE_COUNT = int(os.environ.get("MAX_IDLE_COUNT", torch.iinfo(torch.int64).max))

DEFAULT_EXPLORATION_TYPE: ExplorationType = ExplorationType.RANDOM

_is_osx = sys.platform.startswith("darwin")


class _Interruptor:
    """A class for managing the collection state of a process.

    This class provides methods to start and stop collection, and to check
    whether collection has been stopped. The collection state is protected
    by a lock to ensure thread-safety.
    """

    # interrupter vs interruptor: google trends seems to indicate that "or" is more
    # widely used than "er" even if my IDE complains about that...
    def __init__(self):
        self._collect = True
        self._lock = mp.Lock()

    def start_collection(self):
        with self._lock:
            self._collect = True

    def stop_collection(self):
        with self._lock:
            self._collect = False

    def collection_stopped(self):
        with self._lock:
            return self._collect is False


class _InterruptorManager(SyncManager):
    """A custom SyncManager for managing the collection state of a process.

    This class extends the SyncManager class and allows to share an Interruptor object
    between processes.
    """


_InterruptorManager.register("_Interruptor", _Interruptor)
