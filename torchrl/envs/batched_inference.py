# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import bisect
from collections.abc import Callable, Sequence

import torch
from tensordict import TensorDictBase
from tensordict.tensorclass import NonTensorData, NonTensorStack


__all__ = ["FixedBatchedInference"]


class FixedBatchedInference:
    """Fixed-shape batched inference helper for :class:~torchrl.envs.AsyncEnvPool.

    Accepts variable-size observation batches from
    :meth:~torchrl.envs.AsyncEnvPool.async_step_and_maybe_reset_recv, pads
    them to a fixed *bucket* size, and runs the policy via dedicated CUDA
    streams with double-buffered pinned-memory staging.  This makes the policy
    forward pass compatible with :func:	orch.compile / CUDA graphs (which
    require fixed input shapes) while keeping the data-plane copy overlapped
    with the previous GPU forward pass.

    The helper is stateless between episodes; it is safe to reuse across
    multiple collection loops.

    Args:
        policy (Callable): a callable that maps a batched
            :class:~tensordict.TensorDictBase to a batched
            :class:~tensordict.TensorDictBase (e.g. a
            :class:~tensordict.nn.TensorDictModule).
            :class:~torch.nn.Module instances are automatically moved to
            *device*.
        device (torch.device or str): the device on which the policy runs.

    Keyword Args:
        bucket_sizes (sequence of int): ascending list of batch sizes that the
            helper may pad to.  The smallest bucket `>= len(batch)` is chosen
            on every call.  Must be non-empty and strictly positive.
            Defaults to `[8, 16, 32, 64, 128]`.
        double_buffer (bool): when `True` (default), two pinned staging
            buffers are maintained per bucket so that the CPU can write
            batch N+1 while the GPU reads batch N.  Set to `False` to use a
            single buffer (simpler, but serialises CPU and GPU).
        add_valid_mask (bool): when `True` (default), a boolean tensor
            `"valid_mask"` is added to the device batch before calling the
            policy.  Rows corresponding to real observations are `True`;
            padding rows are `False`.  The key is stripped from the output
            returned to the caller.
        stream (torch.cuda.Stream or None): CUDA stream to use for the policy
            forward pass (the *compute* stream).  `None` (default) creates a
            fresh dedicated stream on *device*.  A separate internal copy stream
            is always created for the async H2D transfer so that H2D and compute
            can genuinely overlap.  Ignored when *device* is CPU.

    .. note::
        Non-tensor metadata keys (e.g. `"env_index"`) are automatically
        routed around the pinned staging buffer and reattached to the output,
        so they are always preserved.  Nested :class:~tensordict.TensorDictBase
        values are correctly identified as tensor data and included in staging.

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

        self.device = torch.device(device)
        self.bucket_sizes = bucket_sizes
        self.double_buffer = double_buffer
        self.add_valid_mask = add_valid_mask
        self._num_buffers = 2 if double_buffer else 1

        # Auto-move nn.Module policies to the target device.
        if isinstance(policy, torch.nn.Module):
            policy = policy.to(self.device)
        self.policy = policy

        self._staging: dict[int, list[TensorDictBase]] = {}
        # Tracks when it is safe to overwrite each pinned staging buffer
        # (i.e. the H2D copy from that buffer has finished).
        self._copy_events: dict[int, list[torch.cuda.Event | None]] = {}
        self._buf_idx: dict[int, int] = {}

        if self.device.type == "cuda":
            # Separate streams so H2D copy and policy forward can overlap.
            self._copy_stream = torch.cuda.Stream(device=self.device)
            self._compute_stream = stream or torch.cuda.Stream(device=self.device)
        else:
            self._copy_stream = None
            self._compute_stream = None

        self._initialized = False

    @staticmethod
    def _non_tensor_keys(batch: TensorDictBase) -> list[str]:
        """Return top-level keys whose values are not tensors or TensorDicts.

        Non-tensor metadata (e.g. `NonTensorStack` env_index) cannot live in
        pinned staging buffers and must be routed around them.

        Note: we inspect the **top-level value** (not recursive leaves) so that
        nested TensorDicts — which contain tensor leaves reachable via tuple
        keys — are correctly classified as tensor data, not metadata.
        """
        return [
            k
            for k in batch.keys()
            if isinstance(batch.get(k), (NonTensorStack, NonTensorData))
        ]

    def _init_from_batch(self, batch: TensorDictBase) -> None:
        """Allocate pinned staging buffers per bucket from the first batch."""
        # Only tensor leaves and nested TensorDicts go into staging.
        # Non-tensor metadata (e.g. env_index) is routed separately in __call__.
        meta_keys = self._non_tensor_keys(batch)
        tensor_keys = [k for k in batch.keys() if k not in meta_keys]
        tensor_template = batch.select(*tensor_keys)[:1]

        for bucket in self.bucket_sizes:
            template = tensor_template.expand(bucket).clone()

            # Pre-allocate valid_mask inside the template so it is covered by
            # pin_memory() and copied to device in the same non-blocking call.
            if self.add_valid_mask:
                template.set(self._MASK_KEY, torch.zeros(bucket, dtype=torch.bool))

            buffers: list[TensorDictBase] = []
            events: list = []
            for _ in range(self._num_buffers):
                if self.device.type == "cuda":
                    buf = template.clone().pin_memory()
                    ev = torch.cuda.Event()
                    # Pre-record so the first event.synchronize() is a no-op.
                    ev.record(self._copy_stream)
                else:
                    buf = template.clone()
                    ev = None
                buffers.append(buf)
                events.append(ev)
            self._staging[bucket] = buffers
            self._copy_events[bucket] = events
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
            batch (TensorDictBase): a 1-D batch of shape `[B]`.

        Returns:
            TensorDictBase: the policy output on *device*, shape `[B]`.
            Non-tensor metadata keys (e.g. `"env_index"`) from the input are
            reattached to the output.  `valid_mask` is stripped.
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

        # Separate non-tensor metadata from tensor / nested-TensorDict keys.
        # Nested TensorDicts are identified by checking the top-level value type
        # directly; iterating recursive leaves would yield tuple keys that do not
        # appear in batch.keys(), causing nested TDs to be misclassified.
        non_tensor_keys = self._non_tensor_keys(batch)
        tensor_keys = [k for k in batch.keys() if k not in non_tensor_keys]

        bucket = self._pick_bucket(B)
        buf_idx = self._buf_idx[bucket]
        staging = self._staging[bucket][buf_idx]
        copy_event = self._copy_events[bucket][buf_idx]

        # Block until the H2D copy that read from this staging buffer is done,
        # so it is safe to overwrite the pinned CPU memory.
        if copy_event is not None:
            copy_event.synchronize()

        # Fill valid rows in-place with tensor leaves only (zero allocation).
        staging[:B].update_(batch.select(*tensor_keys))
        if B < bucket:
            staging[B:].zero_()

        # valid_mask lives in pinned memory; update it in-place before H2D.
        if self.add_valid_mask:
            mask = staging.get(self._MASK_KEY)
            mask.fill_(False)
            mask[:B] = True

        if self._copy_stream is not None:
            # Async H2D on the copy stream.
            with torch.cuda.stream(self._copy_stream):
                device_batch = staging.to(self.device, non_blocking=True)

            # Record after the copy so we know when the buffer is safe to reuse.
            new_copy_event = torch.cuda.Event()
            new_copy_event.record(self._copy_stream)
            self._copy_events[bucket][buf_idx] = new_copy_event

            # Compute stream waits for H2D to finish, then runs the policy.
            # This separation allows the *next* H2D (double-buffer) to overlap
            # with compute for the current batch.
            self._compute_stream.wait_stream(self._copy_stream)
            with torch.cuda.stream(self._compute_stream):
                # Tell the caching allocator that device_batch tensors are live
                # on _compute_stream.  Without this, if device_batch goes out of
                # scope before the stream reaches the forward pass, the allocator
                # may reuse the backing memory while the GPU is still reading it.
                device_batch.apply(
                    lambda t: t.record_stream(torch.cuda.current_stream())
                )
                output = self.policy(device_batch)

            # Make the *calling* stream wait for compute before the caller reads
            # the result tensors — without this, reads on the default stream can
            # race with the forward pass on _compute_stream.
            torch.cuda.current_stream().wait_stream(self._compute_stream)
        else:
            device_batch = staging.to(self.device)
            output = self.policy(device_batch)

        self._buf_idx[bucket] = (buf_idx + 1) % self._num_buffers

        result = output[:B]
        if self.add_valid_mask and self._MASK_KEY in result.keys():
            result = result.exclude(self._MASK_KEY)

        # Reattach non-tensor metadata from the original batch.
        for k in non_tensor_keys:
            result.set(k, batch.get(k))

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
        self._copy_events.clear()
        self._buf_idx.clear()
        self._initialized = False
