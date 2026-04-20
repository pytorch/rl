# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util

from queue import Full as QueueFull, Queue

from tensordict import TensorDictBase

from torchrl._utils import logger as torchrl_logger

_has_ray = importlib.util.find_spec("ray") is not None


class _QueueAsRB:
    def __init__(self, queue: Queue | ray.util.queue.Queue):  # noqa
        if not _has_ray:
            raise ImportError("Ray not installed.")
        self.queue = queue

    def extend(self, data: TensorDictBase):
        from ray.util.queue import Full as RayQueueFull

        # unbind the data and put in the queue
        for item in data.unbind(0):
            while True:
                try:
                    self.queue.put_nowait(item)
                    break
                except (QueueFull, RayQueueFull):
                    self.queue.get()  # Remove the oldest item to make space
                    torchrl_logger.warn("rollout queue full. Discarding data.")
        return
