# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import multiprocessing as mp
import time
from typing import Generic, TypeVar

T = TypeVar("T")


class SharedBlock(Generic[T]):
    """Versioned shared state plus a process-safe doorbell.

    The wrapped value is expected to already use shared storage when needed.
    TensorDict and tensor-like values are updated in place when they expose an
    ``update_`` or ``copy_`` method.
    """

    def __init__(
        self,
        value: T,
        *,
        context: mp.context.BaseContext | None = None,
    ) -> None:
        context = context if context is not None else mp.get_context()
        self._value = value
        self._version = context.Value("q", 0)
        self._condition = context.Condition()

    @property
    def version(self) -> int:
        """Current monotonically increasing version."""
        return int(self._version.value)

    @property
    def value(self) -> T:
        """Return the shared value without copying it."""
        return self._value

    def publish(self, value: T) -> int:
        """Update the shared value in place and notify waiting readers."""
        with self._condition:
            if hasattr(self._value, "update_"):
                self._value.update_(value)
            elif hasattr(self._value, "copy_"):
                self._value.copy_(value)
            else:
                self._value = value
            self._version.value += 1
            version = int(self._version.value)
            self._condition.notify_all()
            return version

    def wait(
        self, after_version: int, timeout: float | None = None
    ) -> tuple[T, int] | None:
        """Wait for a version newer than ``after_version``."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._version.value <= after_version:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._condition.wait(timeout=remaining)
            return self._value, int(self._version.value)
