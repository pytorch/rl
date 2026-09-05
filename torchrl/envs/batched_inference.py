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
from tensordict.utils import NestedKey


__all__ = ["FixedBatchedInference"]


class FixedBatchedInference:
    """Fixed-shape batched inference helper for :class:`~torchrl.envs.AsyncEnvPool`.

    Accepts variable-size observation batches from
    :meth:`~torchrl.envs.AsyncEnvPool.async_step_and_maybe_reset_recv`, pads
    them to a fixed *bucket* size, and runs the policy on a dedicated CUDA
    stream. Staged inputs travel through double-buffered pinned staging and a
    single asynchronous H2D copy into a *persistent* per-bucket device buffer,
    so the policy always sees the same shapes and, with
    ``double_buffer=False``, the same device pointers -- making it compatible
    with :func:`torch.compile` (stable shapes avoid recompilation) and with
    manual CUDA-graph capture (stable input storage).

    Only the keys the policy reads are staged and transferred: the keys are
    taken from ``select_keys`` when provided, from ``policy.in_keys`` when the
    policy exposes it (e.g. :class:`~tensordict.nn.TensorDictModule`), and
    fall back to every tensor key of the first batch otherwise.

    The helper retains reusable staging buffers but no episode-specific state,
    so it is safe to reuse across collection loops.

    Args:
        policy (Callable): a callable that maps a batched
            :class:`~tensordict.TensorDictBase` to a batched
            :class:`~tensordict.TensorDictBase` (e.g. a
            :class:`~tensordict.nn.TensorDictModule`). The caller is
            responsible for placing module policies on their intended device
            or devices.
        device (torch.device or str): the device used for staged policy inputs
            and CUDA streams. For policies spanning multiple devices, this is
            the device on which the policy expects its input.

    Keyword Args:
        bucket_sizes (sequence of int): ascending list of batch sizes that the
            helper may pad to.  The smallest bucket ``>= len(batch)`` is chosen
            on every call.  Must be non-empty and strictly positive. The
            largest bucket must cover the pool's ``max_get``. Defaults to
            ``[8, 16, 32, 64, 128]``.
        select_keys (sequence of NestedKey, optional): the keys to stage and
            transfer to the device. Defaults to ``policy.in_keys`` when
            available, otherwise all tensor keys of the first batch.
        double_buffer (bool): when ``True`` (default), two pinned staging
            buffers and two persistent device buffers are maintained per
            bucket so that the CPU can write batch N+1 while the GPU reads
            batch N. Set to ``False`` to keep a single buffer per bucket:
            CPU and GPU serialize, but every call then reuses the same device
            storage, which is what manual CUDA-graph capture of the policy
            requires.
        add_valid_mask (bool): when ``True`` (default), a boolean tensor
            ``"valid_mask"`` is added to the device batch before calling the
            policy.  Rows corresponding to real observations are ``True``;
            padding rows are ``False``.  The key is stripped from the output
            returned to the caller.
        stream (torch.cuda.Stream or None): CUDA stream to use for the policy
            forward pass (the *compute* stream).  ``None`` (default) creates a
            fresh dedicated stream on *device*.  A separate internal copy stream
            is always created for the async H2D transfer.  Ignored when
            *device* is CPU.

    .. note::
        The returned tensordict carries the policy outputs and the non-tensor
        metadata of the input batch (e.g. ``"env_index"``, which is routed
        around the staging buffers and reattached). Staged input keys are not
        echoed back: on the CUDA path they are views of a reused device
        buffer, so echoing them would hand the caller rows that a later call
        overwrites.

    .. note::
        In the synchronous ``recv -> helper -> send`` loop the caller
        typically synchronizes to bring actions back to the CPU before
        sending, so double buffering mainly overlaps the CPU staging write
        with the previous forward pass; full H2D/compute/D2H overlap requires
        a pipelined consumer.

    .. note::
        Initialisation is *lazy*: staging and device buffers and CUDA events
        are allocated from the first incoming batch, so no spec information
        needs to be provided up front.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.envs import FixedBatchedInference
        >>> policy = TensorDictModule(
        ...     torch.nn.Linear(4, 2),
        ...     in_keys=["observation"],
        ...     out_keys=["action"],
        ... )
        >>> helper = FixedBatchedInference(
        ...     policy, device="cpu", bucket_sizes=[8]
        ... )
        >>> batch = TensorDict(
        ...     {"observation": torch.randn(3, 4)}, batch_size=[3]
        ... )
        >>> result = helper(batch)
        >>> assert result.batch_size == torch.Size([3])
        >>> assert result["action"].shape == torch.Size([3, 2])
    """

    _MASK_KEY = "valid_mask"

    def __init__(
        self,
        policy: Callable,
        device: torch.device | str,
        *,
        bucket_sizes: Sequence[int] = (8, 16, 32, 64, 128),
        select_keys: Sequence[NestedKey] | None = None,
        double_buffer: bool = True,
        add_valid_mask: bool = True,
        stream: torch.cuda.Stream | None = None,
    ):
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
        self._select_keys = list(select_keys) if select_keys is not None else None

        self._num_buffers = 2 if double_buffer else 1

        self._staging: dict[int, list[TensorDictBase]] = {}
        self._device_batches: dict[int, list[TensorDictBase]] = {}
        self._copy_events: dict[int, list[torch.cuda.Event | None]] = {}
        self._compute_events: dict[int, list[torch.cuda.Event | None]] = {}
        self._dirty_rows: dict[int, list[int]] = {}
        self._buf_idx: dict[int, int] = {}
        self._staging_keys: list[NestedKey] = []

        if self.device.type == "cuda":
            self._copy_stream = torch.cuda.Stream(device=self.device)
            self._compute_stream = stream or torch.cuda.Stream(device=self.device)
        else:
            self._copy_stream = None
            self._compute_stream = None

        self._initialized = False

    @staticmethod
    def _non_tensor_keys(batch: TensorDictBase) -> list[str]:
        return [
            key
            for key in batch.keys()
            if isinstance(batch.get(key), (NonTensorStack, NonTensorData))
        ]

    def _resolve_staging_keys(self, batch: TensorDictBase) -> list[NestedKey]:
        if self._select_keys is not None:
            keys = self._select_keys
        else:
            keys = getattr(self.policy, "in_keys", None)
            if keys is None:
                non_tensor_keys = self._non_tensor_keys(batch)
                keys = [key for key in batch.keys() if key not in non_tensor_keys]
        # Deduplicate while preserving order; the mask is staged separately.
        return [key for key in dict.fromkeys(keys) if key != self._MASK_KEY]

    def _init_from_batch(self, batch: TensorDictBase) -> None:
        """Allocate staging and device buffers per bucket size from the first batch."""
        self._staging_keys = self._resolve_staging_keys(batch)
        tensor_template = batch.select(*self._staging_keys)[:1]
        for bucket in self.bucket_sizes:
            template = tensor_template.expand(bucket).clone()
            if self.add_valid_mask:
                template.set(self._MASK_KEY, torch.zeros(bucket, dtype=torch.bool))

            buffers: list[TensorDictBase] = []
            device_batches: list[TensorDictBase] = []
            copy_events: list = []
            compute_events: list = []
            for _ in range(self._num_buffers):
                if self.device.type == "cuda":
                    buf = template.clone().pin_memory()
                    device_batches.append(template.to(self.device))
                    copy_event = torch.cuda.Event()
                    compute_event = torch.cuda.Event()
                    # Pre-record so the first waits are no-ops.
                    copy_event.record(self._copy_stream)
                    compute_event.record(self._compute_stream)
                else:
                    buf = template.clone()
                    copy_event = None
                    compute_event = None
                buffers.append(buf)
                copy_events.append(copy_event)
                compute_events.append(compute_event)

            self._staging[bucket] = buffers
            self._device_batches[bucket] = device_batches
            self._copy_events[bucket] = copy_events
            self._compute_events[bucket] = compute_events
            # Template rows replicate the first batch row, so the first use of
            # each buffer must zero the full padding region.
            self._dirty_rows[bucket] = [bucket] * self._num_buffers
            self._buf_idx[bucket] = 0

        if self.device.type == "cuda":
            init_event = torch.cuda.Event()
            init_event.record(torch.cuda.current_stream(self.device))
            self._copy_stream.wait_event(init_event)
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
            TensorDictBase: the policy outputs on *device*, shape ``[B]``,
            with the input batch's non-tensor metadata reattached.
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

        batch_keys = set(batch.keys(True))
        missing = [key for key in self._staging_keys if key not in batch_keys]
        if missing:
            raise ValueError(
                f"Tensor keys changed after staging initialisation: the staged "
                f"keys {sorted(map(str, self._staging_keys))} are no longer all "
                f"present (missing {sorted(map(str, missing))})."
            )

        bucket = self._pick_bucket(B)
        buf_idx = self._buf_idx[bucket]
        staging = self._staging[bucket][buf_idx]
        copy_event = self._copy_events[bucket][buf_idx]

        if copy_event is not None:
            # Host-side guard: the previous H2D copy out of this staging
            # buffer must have completed before the CPU overwrites it.
            copy_event.synchronize()

        staging[:B].update_(batch.select(*self._staging_keys))

        dirty = self._dirty_rows[bucket][buf_idx]
        if B < dirty:
            staging[B:dirty].zero_()
        self._dirty_rows[bucket][buf_idx] = B

        if self.add_valid_mask:
            mask = staging.get(self._MASK_KEY)
            mask.fill_(False)
            mask[:B] = True

        if self._copy_stream is not None:
            device_batch = self._device_batches[bucket][buf_idx]
            compute_event = self._compute_events[bucket][buf_idx]
            with torch.cuda.stream(self._copy_stream):
                # Device-side guard: the previous forward pass reading this
                # persistent device buffer must be done before it is
                # overwritten. The wait is enqueued on the copy stream; the
                # host does not block.
                self._copy_stream.wait_event(compute_event)
                device_batch.copy_(staging, non_blocking=True)
            copy_event.record(self._copy_stream)

            self._compute_stream.wait_stream(self._copy_stream)
            with torch.cuda.stream(self._compute_stream):
                output = self.policy(device_batch)
        else:
            if self.device.type == "cpu":
                device_batch = staging.clone()
            else:
                device_batch = staging.to(self.device)
            output = self.policy(device_batch)

        self._buf_idx[bucket] = (buf_idx + 1) % self._num_buffers

        result = output[:B]
        exclude_keys = [key for key in self._staging_keys if key in result.keys(True)]
        if self.add_valid_mask and self._MASK_KEY in result.keys():
            exclude_keys.append(self._MASK_KEY)
        if exclude_keys:
            result = result.exclude(*exclude_keys)
        if self._copy_stream is not None:
            device_storage_ptrs = {
                value.untyped_storage().data_ptr()
                for value in device_batch.values(True, True)
                if isinstance(value, torch.Tensor) and value.layout == torch.strided
            }
            with torch.cuda.stream(self._compute_stream):
                if any(
                    isinstance(value, torch.Tensor)
                    and (
                        value.layout != torch.strided
                        or value.untyped_storage().data_ptr() in device_storage_ptrs
                    )
                    for value in result.values(True, True)
                ):
                    # Policies may return aliases or views of their inputs.
                    # Detach those outputs before the persistent input buffer
                    # is made available to the next call.
                    result = result.clone()
            compute_event.record(self._compute_stream)

            caller_stream = torch.cuda.current_stream(self.device)
            caller_stream.wait_stream(self._compute_stream)
            result.record_stream(caller_stream)

        for key in self._non_tensor_keys(batch):
            result.set(key, batch.get(key))

        return result

    def __enter__(self) -> FixedBatchedInference:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        """Release staging and device buffers and CUDA events.

        Call this (or use the helper as a context manager) when the collection
        loop finishes to free pinned and device memory promptly.
        """
        if self._compute_stream is not None:
            self._compute_stream.synchronize()
            self._copy_stream.synchronize()
        self._staging.clear()
        self._device_batches.clear()
        self._copy_events.clear()
        self._compute_events.clear()
        self._dirty_rows.clear()
        self._buf_idx.clear()
        self._staging_keys = []
        self._initialized = False
