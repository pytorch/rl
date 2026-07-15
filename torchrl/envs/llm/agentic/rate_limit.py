# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Asyncio rate limiter -- semaphore + token bucket.

A :class:`RateLimiter` caps concurrent in-flight calls and (optionally) the
sustained call rate. Used by :class:`~torchrl.envs.llm.agentic.ToolCompose`
to throttle individual tools (e.g. a search API at 5 QPS) without blocking
the rest of the dispatch.
"""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager


class RateLimiter:
    """Combined semaphore + token-bucket throttle.

    Args:
        max_concurrent: Cap on simultaneously in-flight calls. ``None`` for
            unlimited.
        rate_per_second: Sustained refill rate. ``None`` disables the
            token bucket; only the semaphore is enforced.
        burst: Token-bucket capacity. Defaults to ``rate_per_second``.

    Examples:
        >>> import asyncio
        >>> async def go():
        ...     limiter = RateLimiter(max_concurrent=2)
        ...     async with limiter.slot():
        ...         pass
        >>> asyncio.run(go())
    """

    def __init__(
        self,
        *,
        max_concurrent: int | None = None,
        rate_per_second: float | None = None,
        burst: float | None = None,
    ) -> None:
        self._sem: asyncio.Semaphore | None = (
            asyncio.Semaphore(max_concurrent) if max_concurrent else None
        )
        self._rate = rate_per_second
        self._capacity = burst if burst is not None else (rate_per_second or 0.0)
        self._tokens = self._capacity
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def _consume(self) -> None:
        if not self._rate:
            return
        async with self._lock:
            now = time.monotonic()
            self._tokens = min(
                self._capacity, self._tokens + (now - self._last) * self._rate
            )
            self._last = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            wait = (1.0 - self._tokens) / self._rate
            # Reserve this future token while holding the lock. Without this,
            # all concurrent waiters observe the same deficit, sleep together,
            # and are released as an unbounded burst.
            self._tokens -= 1.0
        await asyncio.sleep(wait)

    @asynccontextmanager
    async def slot(self):
        """Acquire one slot.

        Blocks until both the semaphore and token bucket allow.
        """
        await self._consume()
        if self._sem is None:
            yield
            return
        async with self._sem:
            yield


__all__ = ["RateLimiter"]
