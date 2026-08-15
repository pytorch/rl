# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import bisect
from collections.abc import Callable, Sequence

import torch
from tensordict import TensorDictBase


__all__ = ["FixedBatchedInference"]


class FixedBatchedInference:
    """Fixed-shape batched inference helper for :class:`~torchrl.envs.AsyncEnvPool`.

    Accepts variable-size observation batches from
    :meth:`~torchrl.envs.AsyncEnvPool.async_step_and_maybe_reset_recv`, pads
    them to a fixed *bucket* size, and runs the policy via a dedicated CUDA
    stream with double-buffered pinned-memory staging.  This makes the policy
    forward pass compatible with :func:`torch.compile` / CUDA graphs (which
    require fixed input shapes) while keeping the data-plane copy overlapped
    with the previous GPU forward pass.

    The helper is stateless between episodes; it is safe to reuse across
    multiple collection loops.

    Args:
        policy (Callable): a callable that maps a batched
            :class:`~tensordict.TensorDictBase` to a batched
            :class:`~tensordict.TensorDictBase` (e.g. a
            :class:`~tensordict.nn.TensorDictModule`).
        device (torch.device or str): the device on which the policy runs.

    Keyword Args:
        bucket_sizes (sequence of int): ascending list of batch sizes that the
            helper may pad to.  The smallest bucket ``>= len(batch)`` is chosen
            on every call.  Must be non-empty and strictly positive.
            Defaults to ``[8, 16, 32, 64, 128]``.
        double_buffer (bool): when ``True`` (default), two pinned staging
            buffers are maintained per bucket so that the CPU can write
            batch N+1 while the GPU reads batch N.  Set to ``False`` to use a
            single buffer (simpler, but serialises CPU and GPU).
        add_valid_mask (bool): when ``True`` (default), a boolean tensor
            ``"valid_mask"`` is added to the device batch before calling the
            policy.  Rows corresponding to real observations are ``True``;
            padding rows are ``False``.  The key is stripped from the output
            returned to the caller.
        stream (torch.cuda.Stream or None): CUDA stream for the async H2D
            transfer and the policy forward pass.  ``None`` (default) creates
            a fresh dedicated stream on *device*.  Ignored when *device* is CPU.

    .. note::
        Initialisation is *lazy*: pinned staging buffers and CUDA events are
        allocated from the first incoming batch, so no spec information needs
        to be provided up front.

    Example:
        >>> from functools import partial
        >>> import torch
        >>> import torch.nn as nn
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.envs import AsyncEnvPool, GymEnv
        >>> from torchrl.envs.batched_inference import FixedBatchedInference
        >>> policy = TensorDictModule(
        ...     nn.Linear(4, 2), in_keys=["observation"], out_keys=["action"]
        ... )
        >>> helper = FixedBatchedInference(
        ...     policy, device="cuda:0", bucket_sizes=[8, 16, 32, 64]
        ... )
        >>> pool = AsyncEnvPool(
        ...     [partial(GymEnv, "CartPole-v1")] * 16,
        ...     backend="multiprocessing",
        ...     exchange="shm",
        ... )
    """

    _MASK_KEY = "valid_mask"

    def __init__(
        self,
        policy: Callable,
        device: torch.device | str,
        *,
        bucket_sizes: Sequence[int] = (8, 16, 32, 64, 128),
        double_buffer: bool = True,
        add_valid_mask: bool = True,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        bucket_sizes = sorted(set(bucket_sizes))
        if not bucket_sizes:
            raise ValueError("bucket_sizes must not be empty.")
        if bucket_sizes[0] < 1:
            raise ValueError("All bucket sizes must be >= 1.")

        self.policy = policy
        self.device = torch.device(device)
        self.bucket_sizes = bucket_sizes
        self.double_buffer = double_buffer
        self.add_valid_mask = add_valid_mask

        self._num_buffers = 2 if double_buffer else 1

        self._staging: dict[int, list[TensorDictBase]] = {}
        self._events: dict[int, list[torch.cuda.Event | None]] = {}
        self._buf_idx: dict[int, int] = {}

        if self.device.type == "cuda":
            self._stream = stream or torch.cuda.Stream(device=self.device)
        else:
            self._stream = None

        self._initialized = False

    def _init_from_batch(self, batch: TensorDictBase) -> None:
        """Allocate pinned staging buffers per bucket size from the first batch."""
        for bucket in self.bucket_sizes:
            template = batch[:1].expand(bucket).clone()

            buffers: list[TensorDictBase] = []
            events: list = []
            for _ in range(self._num_buffers):
                if self.device.type == "cuda":
                    buf = template.clone().pin_memory()
                    ev = torch.cuda.Event()
                    # Pre-record so the first event.synchronize() is a no-op.
                    ev.record(self._stream)
                else:
                    buf = template.clone()
                    ev = None
                buffers.append(buf)
                events.append(ev)

            self._staging[bucket] = buffers
            self._events[bucket] = events
            self._buf_idx[bucket] = 0

        self._initialized = True

    def _pick_bucket(self, batch_size: int) -> int:
        idx = bisect.bisect_left(self.bucket_sizes, batch_size)
        if idx >= len(self.bucket_sizes):
            raise ValueError(
                f"Incoming batch size {batch_size} exceeds the largest bucket "
                f"({self.bucket_sizes[-1]}). Add a larger entry to bucket_sizes."
            )
        return self.bucket_sizes[idx]

    @torch.no_grad()
    def __call__(self, batch: TensorDictBase) -> TensorDictBase:
        """Pad *batch*, copy to device asynchronously, run the policy.

        Args:
            batch (TensorDictBase): a 1-D batch of shape ``[B]``.

        Returns:
            TensorDictBase: the policy output on *device*, shape ``[B]``.
        """
        if batch.batch_dims != 1:
            raise ValueError(
                f"FixedBatchedInference expects a 1-D TensorDict, "
                f"got batch_dims={batch.batch_dims}."
            )

        B = batch.batch_size[0]
        if B == 0:
            raise ValueError("Received an empty batch (B=0).")

        if not self._initialized:
            self._init_from_batch(batch)

        bucket = self._pick_bucket(B)
        buf_idx = self._buf_idx[bucket]
        staging = self._staging[bucket][buf_idx]
        event = self._events[bucket][buf_idx]

        if event is not None:
            event.synchronize()

        staging[:B].update_(batch)

        if B < bucket:
            staging[B:].zero_()

        if self.add_valid_mask:
            mask = torch.zeros(bucket, dtype=torch.bool)
            mask[:B] = True
            staging.set(self._MASK_KEY, mask, inplace=False)

        if self._stream is not None:
            with torch.cuda.stream(self._stream):
                device_batch = staging.to(self.device, non_blocking=True)
                new_event = torch.cuda.Event()
                new_event.record(self._stream)
                self._events[bucket][buf_idx] = new_event
                output = self.policy(device_batch)
        else:
            device_batch = staging.to(self.device)
            output = self.policy(device_batch)

        self._buf_idx[bucket] = (buf_idx + 1) % self._num_buffers

        result = output[:B]
        if self.add_valid_mask and self._MASK_KEY in result.keys():
            result = result.exclude(self._MASK_KEY)

        return result

    def __enter__(self) -> FixedBatchedInference:
        return self

    def __exit__(self, *exc_info) -> None:
        self.reset()

    def reset(self) -> None:
        """Release staging buffers and CUDA events.

        Call this (or use the helper as a context manager) when the collection
        loop finishes to free pinned memory promptly.
        """
        self._staging.clear()
        self._events.clear()
        self._buf_idx.clear()
        self._initialized = False
