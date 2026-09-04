# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import json
import warnings
from copy import deepcopy
from multiprocessing.context import get_spawning_popen
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pyvers import implement_for
from tensordict import MemoryMappedTensor
from torch.utils._pytree import tree_map
from torchrl._extension import EXTENSION_WARNING
from torchrl._utils import logger, rl_warnings
from torchrl.data.replay_buffers.storages import Storage, TensorStorage
from torchrl.data.replay_buffers.utils import _is_int, unravel_index

try:
    from torchrl._torchrl import (
        MinSegmentTreeFp32,
        MinSegmentTreeFp64,
        SumSegmentTreeFp32,
        SumSegmentTreeFp64,
    )
except ImportError:
    # Make default values
    MinSegmentTreeFp32 = None
    MinSegmentTreeFp64 = None
    SumSegmentTreeFp32 = None
    SumSegmentTreeFp64 = None

try:
    from torchrl._torchrl import (
        CudaMinSegmentTreeFp32,
        CudaMinSegmentTreeFp64,
        CudaSumSegmentTreeFp32,
        CudaSumSegmentTreeFp64,
    )
except ImportError:
    CudaMinSegmentTreeFp32 = None
    CudaMinSegmentTreeFp64 = None
    CudaSumSegmentTreeFp32 = None
    CudaSumSegmentTreeFp64 = None

_EMPTY_STORAGE_ERROR = "Cannot sample from an empty storage."


# Maps a "with replacement" sampler class to its "without replacement" counterpart.
# Populated at module import time after the relevant classes are defined.
# Consumed by :class:`_SamplerMeta` to dispatch ``Cls(replacement=False, ...)`` calls
# to ``_REPLACEMENT_DISPATCH[Cls](...)``.
_REPLACEMENT_DISPATCH: dict[type, type] = {}


from .base import Sampler


class PrioritizedSampler(Sampler):
    r"""Prioritized sampler for replay buffer.

    This sampler implements Prioritized Experience Replay (PER) as presented in
    "Schaul, T.; Quan, J.; Antonoglou, I.; and Silver, D. 2015. Prioritized experience replay."
    (https://arxiv.org/abs/1511.05952)

    **Core Idea**: Instead of sampling experiences uniformly from the replay buffer,
    PER samples experiences with probability proportional to their "importance" - typically
    measured by the magnitude of their temporal-difference (TD) error. This prioritization
    can lead to faster learning by focusing on experiences that are most informative.

    **How it works**:
    1. Each experience is assigned a priority based on its TD error: :math:`p_i = |\delta_i| + \epsilon`
    2. Sampling probability is computed as: :math:`P(i) = \frac{p_i^\alpha}{\sum_j p_j^\alpha}`
    3. Importance sampling weights correct for the bias: :math:`w_i = (N \cdot P(i))^{-\beta}`

    Args:
        max_capacity (int): maximum capacity of the buffer.
        alpha (:obj:`float`): exponent :math:`\alpha` determines how much prioritization is used.
            - :math:`\alpha = 0`: uniform sampling (no prioritization)
            - :math:`\alpha = 1`: full prioritization based on TD error magnitude
            - Typical values: 0.4-0.7 for balanced prioritization
            - Higher :math:`\alpha` means more aggressive prioritization of high-error experiences
        beta (:obj:`float`): importance sampling negative exponent :math:`\beta`.
            - :math:`\beta` controls the correction for the bias introduced by prioritization
            - :math:`\beta = 0`: no correction (biased towards high-priority samples)
            - :math:`\beta = 1`: full correction (unbiased but potentially unstable)
            - Typical values: start at 0.4-0.6 and anneal to 1.0 during training
            - Lower :math:`\beta` early in training provides stability, higher :math:`\beta` later reduces bias
        eps (:obj:`float`, optional): small constant added to priorities to ensure
            no experience has zero priority. This prevents experiences from never
            being sampled. Defaults to 1e-8.
        reduction (str, optional): the reduction method for multidimensional
            tensordicts (ie stored trajectory). Can be one of "max", "min",
            "median" or "mean".
        max_priority_within_buffer (bool, optional): if ``True``, the max-priority
            is tracked within the buffer. When ``False``, the max-priority tracks
            the maximum value since the instantiation of the sampler.
        device (torch.device or str, optional): device that holds the priority
            trees. Defaults to ``None``, in which case CUDA storage selects a CUDA
            tree when the installed TorchRL extension was built with CUDA support,
            and CPU storage keeps the existing CPU tree.
        max_pending (int, optional): maximum number of :meth:`mark_update` calls
            whose priority writes may be deferred before the sampler flushes them
            to the segment trees. The writer calls ``mark_update`` on every
            ``add``/``extend``, so this caps the memory held by the pending list
            for a buffer that is written to but never read from. ``0`` disables
            the deferral and writes on every ``mark_update``. Defaults to ``64``.

    **Parameter Guidelines**:

    - **:math:`\alpha` (alpha)**: Controls how much to prioritize high-error experiences.
      0.4-0.7: Good balance between learning speed and stability.
      1.0: Maximum prioritization (may be unstable).
      0.0: Uniform sampling (no prioritization benefit).

    - **:math:`\beta` (beta)**: Controls importance sampling correction.
      Start at 0.4-0.6 for training stability.
      Anneal to 1.0 over training to reduce bias.
      Lower values = more stable but biased.
      Higher values = less biased but potentially unstable.

    - **:math:`\epsilon`**: Small constant to prevent zero priorities.
      1e-8: Good default value.
      Too small: may cause numerical issues.
      Too large: reduces prioritization effect.

    Examples:
        >>> from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage, PrioritizedSampler
        >>> from tensordict import TensorDict
        >>> rb = ReplayBuffer(storage=LazyTensorStorage(10), sampler=PrioritizedSampler(max_capacity=10, alpha=1.0, beta=1.0))
        >>> priority = torch.tensor([0, 1000])
        >>> data_0 = TensorDict({"reward": 0, "obs": [0], "action": [0], "priority": priority[0]}, [])
        >>> data_1 = TensorDict({"reward": 1, "obs": [1], "action": [2], "priority": priority[1]}, [])
        >>> rb.add(data_0)
        >>> rb.add(data_1)
        >>> rb.update_priority(torch.tensor([0, 1]), priority=priority)
        >>> sample, info = rb.sample(10, return_info=True)
        >>> print(sample)
        TensorDict(
                fields={
                    action: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.int64, is_shared=False),
                    obs: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.int64, is_shared=False),
                    priority: Tensor(shape=torch.Size([10]), device=cpu, dtype=torch.int64, is_shared=False),
                    reward: Tensor(shape=torch.Size([10]), device=cpu, dtype=torch.int64, is_shared=False)},
                batch_size=torch.Size([10]),
                device=cpu,
                is_shared=False)
        >>> print(info)
        {'priority_weight': array([1.e-11, 1.e-11, 1.e-11, 1.e-11, 1.e-11, 1.e-11, 1.e-11, 1.e-11,
               1.e-11, 1.e-11], dtype=float32), 'index': array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])}

    .. note:: Using a :class:`~torchrl.data.replay_buffers.TensorDictReplayBuffer` can smoothen the
        process of updating the priorities:

            >>> from torchrl.data.replay_buffers import TensorDictReplayBuffer as TDRB, LazyTensorStorage, PrioritizedSampler
            >>> from tensordict import TensorDict
            >>> rb = TDRB(
            ...     storage=LazyTensorStorage(10),
            ...     sampler=PrioritizedSampler(max_capacity=10, alpha=1.0, beta=1.0),
            ...     priority_key="priority",  # This kwarg isn't present in regular RBs
            ... )
            >>> priority = torch.tensor([0, 1000])
            >>> data_0 = TensorDict({"reward": 0, "obs": [0], "action": [0], "priority": priority[0]}, [])
            >>> data_1 = TensorDict({"reward": 1, "obs": [1], "action": [2], "priority": priority[1]}, [])
            >>> data = torch.stack([data_0, data_1])
            >>> rb.extend(data)
            >>> rb.update_priority(data)  # Reads the "priority" key as indicated in the constructor
            >>> sample, info = rb.sample(10, return_info=True)
            >>> print(sample['index'])  # The index is packed with the tensordict
            tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    """

    # Version of the state_dict / dumps payload schema. Version 1 marks
    # payloads whose _max_priority is stored in the raw priority domain
    # (pytorch/rl#3925); version-less payloads are treated as pre-#3925.
    _STATE_SCHEMA_VERSION: int = 1

    def __init__(
        self,
        max_capacity: int,
        alpha: float,
        beta: float,
        eps: float = 1e-8,
        dtype: torch.dtype = torch.float,
        reduction: str = "max",
        max_priority_within_buffer: bool = False,
        device: torch.device | str | None = None,
        max_pending: int = 64,
    ) -> None:
        if alpha < 0:
            raise ValueError(
                f"alpha must be greater or equal than 0, got alpha={alpha}"
            )
        if beta < 0:
            raise ValueError(f"beta must be greater or equal to 0, got beta={beta}")
        if max_pending < 0:
            raise ValueError(
                f"max_pending must be greater or equal to 0, got max_pending={max_pending}"
            )

        self._max_pending = max_pending
        self._max_capacity = max_capacity
        self._alpha = alpha
        self._beta = beta
        self._eps = eps
        self.reduction = reduction
        self.dtype = dtype
        self._max_priority_within_buffer = max_priority_within_buffer
        self._device = torch.device(device) if device is not None else None
        self._init()
        if rl_warnings() and SumSegmentTreeFp32 is None:
            logger.warning(EXTENSION_WARNING)

    def __repr__(self):
        return f"{self.__class__.__name__}(alpha={self._alpha}, beta={self._beta}, eps={self._eps}, reduction={self.reduction})"

    @property
    def max_size(self):
        return self._max_capacity

    @property
    def device(self) -> torch.device:
        tree_device = getattr(self._sum_tree, "device", None)
        if tree_device is not None:
            return torch.device(tree_device)
        if self._device is not None:
            return self._device
        return torch.device("cpu")

    @property
    def alpha(self):
        """The priority exponent.

        .. note:: Setting ``alpha`` on a sampler that already holds priorities
          (e.g. when annealing it with a
          :class:`~torchrl.data.replay_buffers.scheduler.ParameterScheduler`)
          re-transforms the ``(p + eps) ** alpha`` values stored in the
          sum/min trees to the new exponent in a single O(capacity) pass, so
          sampling probabilities stay consistent with the new value. The one
          exception is changing ``alpha`` away from exactly ``0``: the raw
          priorities cannot be recovered from the trees in that regime, so the
          stored (uniform) values are kept -- and a warning is emitted --
          until each entry's priority is next updated.
        """
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if value < 0:
            raise ValueError(
                f"alpha must be greater or equal than 0, got alpha={value}"
            )
        self._flush_pending_updates()
        old_alpha = self._alpha
        self._alpha = value
        if value != old_alpha:
            self._retransform_priority_trees(old_alpha, value)

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        self._beta = value

    def __getstate__(self):
        if get_spawning_popen() is not None:
            raise RuntimeError(
                f"Samplers of type {type(self)} cannot be shared between processes. "
                "Use TensorDictPrioritizedReplayBuffer(sync=False) instead: "
                "the writer process gets a uniform sampler and the learner "
                "keeps a local prioritized sampler."
            )
        # Pending entries hold a reference to the storage they were marked
        # against, which must not be dragged into the pickled state.
        self._flush_pending_updates()
        return super().__getstate__()

    def _tree_device_from_storage(self, storage: Storage | None) -> torch.device | None:
        if self._device is not None:
            return self._device
        if storage is None:
            return None
        device = getattr(storage, "device", None)
        if device is None or device == "auto":
            return None
        device = torch.device(device)
        if device.type == "cuda":
            return device
        return None

    def _maybe_init_from_storage(self, storage: Storage | None) -> None:
        device = self._tree_device_from_storage(storage)
        if device is not None and device != self.device:
            self._device = device
            self._init()

    def _init(self) -> None:
        self._pending_updates = []
        if SumSegmentTreeFp32 is None:
            raise RuntimeError(
                "SumSegmentTreeFp32 is not available. See warning above."
            )
        if MinSegmentTreeFp32 is None:
            raise RuntimeError(
                "MinSegmentTreeFp32 is not available. See warning above."
            )
        if SumSegmentTreeFp64 is None:
            raise RuntimeError(
                "SumSegmentTreeFp64 is not available. See warning above."
            )
        if MinSegmentTreeFp64 is None:
            raise RuntimeError(
                "MinSegmentTreeFp64 is not available. See warning above."
            )
        device = self._device
        if device is not None and device.type == "cuda":
            if (
                CudaSumSegmentTreeFp32 is None
                or CudaMinSegmentTreeFp32 is None
                or CudaSumSegmentTreeFp64 is None
                or CudaMinSegmentTreeFp64 is None
            ):
                raise RuntimeError(
                    "CUDA prioritized replay buffers require a TorchRL CUDA wheel. "
                    "Install a TorchRL wheel matching your PyTorch CUDA variant or "
                    "rebuild TorchRL with FORCE_CUDA=1."
                )
            if self.dtype in (torch.float, torch.FloatType, torch.float32):
                self._sum_tree = CudaSumSegmentTreeFp32(self._max_capacity, device)
                self._min_tree = CudaMinSegmentTreeFp32(self._max_capacity, device)
            elif self.dtype in (torch.double, torch.DoubleTensor, torch.float64):
                self._sum_tree = CudaSumSegmentTreeFp64(self._max_capacity, device)
                self._min_tree = CudaMinSegmentTreeFp64(self._max_capacity, device)
            else:
                raise NotImplementedError(
                    f"dtype {self.dtype} not supported by PrioritizedSampler"
                )
            self._max_priority = None
            return
        if self.dtype in (torch.float, torch.FloatType, torch.float32):
            self._sum_tree = SumSegmentTreeFp32(self._max_capacity)
            self._min_tree = MinSegmentTreeFp32(self._max_capacity)
        elif self.dtype in (torch.double, torch.DoubleTensor, torch.float64):
            self._sum_tree = SumSegmentTreeFp64(self._max_capacity)
            self._min_tree = MinSegmentTreeFp64(self._max_capacity)
        else:
            raise NotImplementedError(
                f"dtype {self.dtype} not supported by PrioritizedSampler"
            )
        self._max_priority = None

    def _empty(self) -> None:
        self._init()

    def _flush_pending_updates(self) -> None:
        """Applies the priority updates deferred by :meth:`mark_update`.

        The pending list is cleared before the updates are replayed so that the
        :meth:`update_priority` calls below do not recurse into this method.
        """
        pending_updates = self._pending_updates
        if not pending_updates:
            return
        self._pending_updates = []
        for index, priority, storage in pending_updates:
            self.update_priority(index, priority, storage=storage)

    @property
    def _max_priority(self) -> tuple[float | None, int | None]:
        max_priority_index = self.__dict__.get("_max_priority")
        if max_priority_index is None:
            return (None, None)
        return max_priority_index

    @_max_priority.setter
    def _max_priority(self, value: tuple[float | None, int | None]) -> None:
        self.__dict__["_max_priority"] = value

    def _maybe_erase_max_priority(
        self, index: torch.Tensor | int | slice | tuple
    ) -> None:
        if not self._max_priority_within_buffer:
            return
        max_priority_index = self._max_priority[1]
        if max_priority_index is None:
            return

        def check_index(index=index, max_priority_index=max_priority_index):
            if isinstance(index, torch.Tensor):
                # index can be 1d or 2d
                if index.ndim == 1:
                    is_overwritten = (index == max_priority_index).any()
                else:
                    is_overwritten = (index == max_priority_index).all(-1).any()
            elif isinstance(index, int):
                is_overwritten = index == max_priority_index
            elif isinstance(index, slice):
                # This won't work if called recursively
                is_overwritten = max_priority_index in range(
                    index.indices(self._max_capacity)
                )
            elif isinstance(index, tuple):
                is_overwritten = isinstance(max_priority_index, tuple)
                if is_overwritten:
                    for idx, mpi in zip(index, max_priority_index):
                        is_overwritten &= check_index(idx, mpi)
            else:
                raise TypeError(f"index of type {type(index)} is not recognized.")
            return is_overwritten

        is_overwritten = check_index()
        if isinstance(is_overwritten, torch.Tensor):
            if is_overwritten.device.type == "cuda":
                self._max_priority = None
                return
            is_overwritten = bool(is_overwritten.item())
        if is_overwritten:
            self._max_priority = None

    def _tree_argmax(self) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.device
        indices = torch.arange(self._max_capacity, dtype=torch.long, device=device)
        values = torch.as_tensor(self._sum_tree[indices], device=device)
        return values.max(0)

    def _retransform_priority_trees(self, old_alpha: float, new_alpha: float) -> None:
        """Rewrites the tree entries from ``(p + eps) ** old_alpha`` to ``(p + eps) ** new_alpha``.

        A single O(capacity) pass keeps the sampling probabilities (and the
        within-buffer max-priority recomputation, which inverts tree values
        with the current ``alpha``) consistent when ``alpha`` changes, e.g.
        when annealed by a
        :class:`~torchrl.data.replay_buffers.scheduler.ParameterScheduler`.
        ``_max_priority`` is tracked in the raw domain and needs no update.
        """
        device = self.device
        indices = torch.arange(self._max_capacity, dtype=torch.long, device=device)
        values = torch.as_tensor(self._sum_tree[indices], device=device)
        # Entries that were never written hold the sum-tree neutral value 0
        # and must stay 0; written entries hold (p + eps) ** alpha > 0.
        written = values > 0
        if not written.any():
            return
        if old_alpha == 0:
            # With alpha == 0 every written entry is 1.0, so the raw
            # priorities cannot be recovered from the trees. Keep the stored
            # (uniform) values; they are corrected as entries get their
            # priority updated.
            warnings.warn(
                "Changing alpha away from 0 on a PrioritizedSampler that "
                "already holds priorities cannot recover the raw priorities "
                "from the sum/min trees (alpha == 0 stores 1.0 for every "
                "written entry). Sampling stays uniform for those entries "
                "until their priority is next updated, so annealing away "
                "from exactly 0 is approximate."
            )
            return
        indices = indices[written]
        raw = (values[written].double() ** (1.0 / old_alpha) - self._eps).clamp_min(0)
        new_values = ((raw + self._eps) ** new_alpha).to(self.dtype)
        self._sum_tree[indices] = new_values
        self._min_tree[indices] = new_values

    def _recompute_max_priority_from_tree(self) -> None:
        """Recomputes the raw ``_max_priority`` from the sum-tree entries.

        The sum-tree stores ``(p + eps) ** alpha`` while the tracked max
        priority lives in the raw domain, so the tree max is inverted before
        being stored. Used when restoring a checkpoint saved before the
        raw-domain convention (pytorch/rl#3925), whose persisted
        ``_max_priority`` may hold a transformed tree value.
        """
        if self._alpha == 0:
            # (p + eps) ** 0 == 1 for every written entry: the raw max cannot
            # be recovered from the tree. Keep the restored value.
            return
        maxval, maxidx = self._tree_argmax()
        if maxval <= 0:
            # nothing was ever written to the trees
            self._max_priority = None
            return
        maxval = (maxval ** (1.0 / self._alpha) - self._eps).clamp_min(0)
        self._max_priority = (maxval, maxidx)

    @property
    def default_priority(self) -> float | torch.Tensor:
        # Return the RAW max priority. Every consumer feeds this value back through
        # ``update_priority``, which applies the ``(p + eps) ** alpha`` transform
        # exactly once. Returning an already-transformed value here caused new items
        # to be transformed twice (``((p + eps) ** alpha + eps) ** alpha``), which
        # systematically under-prioritized them (for ``alpha < 1``) and broke PER's
        # "new experience is sampled at least once" guarantee.
        mp = self._max_priority[0]
        if mp is None:
            mp = 1.0
        if isinstance(mp, torch.Tensor):
            mp = mp.to(self.device)
        return mp

    def sample(self, storage: Storage, batch_size: int) -> torch.Tensor:
        self._maybe_init_from_storage(storage)
        self._flush_pending_updates()
        if len(storage) == 0:
            raise RuntimeError(_EMPTY_STORAGE_ERROR)
        tree_device = self.device
        is_cuda = tree_device.type == "cuda"
        if is_cuda:
            left = torch.zeros((), dtype=torch.long, device=tree_device)
            right = torch.full((), len(storage), dtype=torch.long, device=tree_device)
            p_sum = self._sum_tree.query(left, right)
            p_min = self._min_tree.query(left, right)
        else:
            p_sum = self._sum_tree.query(0, len(storage))
            p_min = self._min_tree.query(0, len(storage))

        if not is_cuda:
            if p_sum <= 0:
                raise RuntimeError("non-positive p_sum")
            if p_min <= 0:
                raise RuntimeError("non-positive p_min")
        # For some undefined reason, only np.random works here.
        # All PT attempts fail, even when subsequently transformed into numpy
        if is_cuda:
            mass = torch.rand(batch_size, device=tree_device, generator=self._rng)
            mass = mass * p_sum
        elif self._rng is None:
            mass = np.random.uniform(0.0, p_sum, size=batch_size)
        else:
            mass = torch.rand(batch_size, generator=self._rng) * p_sum

        # mass = torch.zeros(batch_size, dtype=torch.double).uniform_(0.0, p_sum)
        # mass = torch.rand(batch_size).mul_(p_sum)
        index = self._sum_tree.scan_lower_bound(mass)
        index = torch.as_tensor(index)
        if index.device != tree_device:
            index = index.to(tree_device)
        if not index.ndim:
            index = index.unsqueeze(0)
        index.clamp_max_(len(storage) - 1)
        weight = torch.as_tensor(self._sum_tree[index], device=tree_device)
        if not is_cuda:
            # get indices where weight is 0
            zero_weight = weight == 0
            while zero_weight.any():
                index = torch.where(zero_weight, index - 1, index)
                if (index < 0).any():
                    raise RuntimeError("Failed to find a suitable index")
                weight = torch.as_tensor(self._sum_tree[index])
                zero_weight = weight == 0

        # Importance sampling weight formula:
        #   w_i = (p_i / sum(p) * N) ^ (-beta)
        #   weight_i = w_i / max(w)
        #   weight_i = (p_i / sum(p) * N) ^ (-beta) /
        #       ((min(p) / sum(p) * N) ^ (-beta))
        #   weight_i = ((p_i / sum(p) * N) / (min(p) / sum(p) * N)) ^ (-beta)
        #   weight_i = (p_i / min(p)) ^ (-beta)
        # weight = np.power(weight / (p_min + self._eps), -self._beta)
        weight = torch.pow(weight / p_min, -self._beta)
        if storage.ndim > 1:
            index = unravel_index(index, storage.shape)
        return index, {"priority_weight": weight}

    def add(self, index: torch.Tensor | int) -> None:
        super().add(index)
        self._maybe_erase_max_priority(index)

    def extend(self, index: torch.Tensor | tuple) -> None:
        super().extend(index)
        self._maybe_erase_max_priority(index)

    @torch.no_grad()
    def update_priority(
        self,
        index: int | torch.Tensor,
        priority: float | torch.Tensor,
        *,
        storage: TensorStorage | None = None,
    ) -> None:  # noqa: D417
        """Updates the priority of the data pointed by the index.

        Args:
            index (int or torch.Tensor): indexes of the priorities to be
                updated.
            priority (Number or torch.Tensor): new priorities of the
                indexed elements.

        Keyword Args:
            storage (Storage, optional): a storage used to map the Nd index size to
                the 1d size of the sum_tree and min_tree. Only required whenever
                ``index.ndim > 2``.

        """
        self._maybe_init_from_storage(storage)
        self._flush_pending_updates()
        tree_device = self.device
        priority = torch.as_tensor(priority, device=tree_device).detach()
        index = torch.as_tensor(index, dtype=torch.long, device=tree_device)
        # we need to reshape priority if it has more than one element or if it has
        # a different shape than index
        if priority.numel() > 1 and priority.shape != index.shape:
            try:
                priority = priority.reshape(index.shape[:1])
            except Exception as err:
                raise RuntimeError(
                    "priority should be a number or an iterable of the same "
                    f"length as index. Got priority of shape {priority.shape} and index "
                    f"{index.shape}."
                ) from err
        elif priority.numel() <= 1:
            priority = priority.squeeze()

        # MaxValueWriter will set -1 for items in the data that we don't want
        # to update. We therefore have to keep only the non-negative indices.
        if _is_int(index) and not isinstance(index, torch.Tensor):
            if index == -1:
                return
        else:
            if index.ndim == 0:
                index = index.view(1)
                if priority.ndim == 0:
                    priority = priority.view(1)
            if index.ndim > 1:
                if storage is None:
                    raise RuntimeError(
                        "storage should be provided to Sampler.update_priority when the storage has more "
                        "than one dimension."
                    )
                try:
                    shape = storage.shape
                except AttributeError:
                    raise AttributeError(
                        "Could not retrieve the storage shape. If your storage is not a TensorStorage subclass "
                        "or its shape isn't accessible via the shape attribute, submit an issue on GitHub."
                    )
                if tree_device.type == "cuda":
                    multipliers = torch.ones(
                        index.shape[-1], dtype=torch.long, device=tree_device
                    )
                    for dim in range(index.shape[-1] - 2, -1, -1):
                        multipliers[dim] = multipliers[dim + 1] * shape[dim + 1]
                    index = (index * multipliers).sum(-1)
                else:
                    index = torch.as_tensor(
                        np.ravel_multi_index(index.unbind(-1), shape)
                    )
            valid_index = index >= 0
            if tree_device.type == "cuda":
                index = index[valid_index]
                if priority.ndim:
                    priority = priority[valid_index]
                if index.numel() == 0:
                    return
            elif not valid_index.any():
                return
            elif not valid_index.all():
                index = index[valid_index]
                if priority.ndim:
                    priority = priority[valid_index]

        max_p, max_p_idx = priority.max(dim=0)
        cur_max_priority, cur_max_priority_index = self._max_priority
        if cur_max_priority is None:
            cur_max_priority, cur_max_priority_index = self._max_priority = (
                max_p,
                index[max_p_idx] if index.ndim else index,
            )
        elif tree_device.type == "cuda":
            if self._max_priority_within_buffer:
                cur_max_priority, cur_max_priority_index = max_p, (
                    index[max_p_idx] if index.ndim else index
                )
            else:
                cur_max_priority = torch.maximum(
                    max_p, torch.as_tensor(cur_max_priority, device=tree_device)
                )
                self._max_priority = (cur_max_priority, cur_max_priority_index)
        elif max_p > cur_max_priority:
            cur_max_priority, cur_max_priority_index = self._max_priority = (
                max_p,
                index[max_p_idx] if index.ndim else index,
            )
        priority = torch.pow(priority + self._eps, self._alpha)
        self._sum_tree[index] = priority
        self._min_tree[index] = priority
        if self._max_priority_within_buffer and cur_max_priority_index is not None:
            if self._alpha == 0:
                # With alpha == 0 the tree stores (p + eps) ** 0 == 1 for every
                # entry, so the raw priorities of untouched entries cannot be
                # recovered from it. Keep the raw max tracked above instead of
                # storing a transformed value in ``_max_priority``; sampling is
                # uniform in this regime, so a stale max only matters if alpha
                # is raised later (see the ``alpha`` setter).
                return
            if tree_device.type == "cuda":
                maxval, maxidx = self._tree_argmax()
            elif (index == cur_max_priority_index).any():
                maxval, maxidx = self._tree_argmax()
            else:
                return
            # ``maxval`` is read from the sum-tree, which stores (p + eps) ** alpha.
            # Convert it back to the raw priority so ``_max_priority`` is always the
            # raw max, matching the non-recomputed path and what ``default_priority``
            # expects (it re-applies the alpha transform once via ``update_priority``).
            maxval = (maxval ** (1.0 / self._alpha) - self._eps).clamp_min(0)
            self._max_priority = (maxval, maxidx)

    def mark_update(
        self, index: int | torch.Tensor, *, storage: Storage | None = None
    ) -> None:
        """Marks the given indices for a default-priority update.

        The update is **lazy**: the priority trees are not written to
        immediately. Instead the ``(index, default_priority)`` pair is appended
        to an internal pending-updates list and flushed the next time the trees
        are read (e.g. on :meth:`sample`, :meth:`update_priority`,
        :meth:`state_dict` or :meth:`dumps`). This keeps segment-tree writes off
        the hot ``extend`` path, where they are usually superseded by the first
        :meth:`update_priority` call anyway.

        The default priority is read eagerly, at mark time, so that a later
        change of the running max does not retroactively alter what was marked.
        If :meth:`update_priority` is called for the same indices before the
        flush, the pending defaults are applied first and then overwritten by
        the explicit priorities.

        At most ``max_pending`` calls are deferred: a buffer that is written to
        but never read from flushes on its own rather than growing the pending
        list without bound.
        """
        priority = self.default_priority
        if isinstance(index, torch.Tensor):
            index = index.clone()
        self._pending_updates.append((index, priority, storage))
        # ``mark_update`` is called by the writer on every ``add``/``extend``, and
        # the pending list only ever shrinks when it is flushed. Bound it so that a
        # buffer that is written to but never read from cannot accumulate an
        # unbounded number of deferred index tensors.
        if len(self._pending_updates) > self._max_pending:
            self._flush_pending_updates()

    def state_dict(self) -> dict[str, Any]:
        self._flush_pending_updates()
        return {
            "_schema_version": self._STATE_SCHEMA_VERSION,
            "_alpha": self._alpha,
            "_beta": self._beta,
            "_eps": self._eps,
            "_max_priority": self._max_priority,
            "_sum_tree": deepcopy(self._sum_tree),
            "_min_tree": deepcopy(self._min_tree),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # Version-less payloads predate pytorch/rl#3925, whose within-buffer
        # recompute stored the transformed tree value in _max_priority.
        version = state_dict.get("_schema_version", 0)
        self._alpha = state_dict["_alpha"]
        self._beta = state_dict["_beta"]
        self._eps = state_dict["_eps"]
        # deepcopy to decouple the sampler state from the caller's objects
        # (this also clones any tensor held in _max_priority)
        self._max_priority = deepcopy(state_dict["_max_priority"])
        self._sum_tree = deepcopy(state_dict["_sum_tree"])
        self._min_tree = deepcopy(state_dict["_min_tree"])
        self._pending_updates = []
        if (
            version < 1
            and self._max_priority_within_buffer
            and self._max_priority[0] is not None
        ):
            self._recompute_max_priority_from_tree()

    @implement_for("torch", None, "2.5.0")
    def dumps(self, path):
        raise NotImplementedError("This method is not implemented for Torch < 2.5.0")

    @implement_for("torch", "2.5.0", None)
    def dumps(self, path):  # noqa: F811
        self._flush_pending_updates()
        path = Path(path).absolute()
        path.mkdir(exist_ok=True)
        try:
            mm_st = MemoryMappedTensor.from_filename(
                shape=(self._max_capacity,),
                dtype=torch.float64,
                filename=path / "sumtree.memmap",
            )
            mm_mt = MemoryMappedTensor.from_filename(
                shape=(self._max_capacity,),
                dtype=torch.float64,
                filename=path / "mintree.memmap",
            )
        except FileNotFoundError:
            mm_st = MemoryMappedTensor.empty(
                (self._max_capacity,),
                dtype=torch.float64,
                filename=path / "sumtree.memmap",
            )
            mm_mt = MemoryMappedTensor.empty(
                (self._max_capacity,),
                dtype=torch.float64,
                filename=path / "mintree.memmap",
            )
        mm_st.copy_(
            torch.as_tensor([self._sum_tree[i] for i in range(self._max_capacity)])
        )
        mm_mt.copy_(
            torch.as_tensor([self._min_tree[i] for i in range(self._max_capacity)])
        )
        metadata = tree_map(
            float,
            {
                "_alpha": self._alpha,
                "_beta": self._beta,
                "_eps": self._eps,
                "_max_priority": self._max_priority,
                "_max_capacity": self._max_capacity,
            },
        )
        metadata["_schema_version"] = self._STATE_SCHEMA_VERSION
        with open(path / "sampler_metadata.json", "w") as file:
            json.dump(metadata, file)

    @implement_for("torch", None, "2.5.0")
    def loads(self, path):
        raise NotImplementedError("This method is not implemented for Torch < 2.5.0")

    @implement_for("torch", "2.5.0", None)
    def loads(self, path):  # noqa: F811
        path = Path(path).absolute()
        with open(path / "sampler_metadata.json") as file:
            metadata = json.load(file)
        # Version-less payloads predate pytorch/rl#3925, whose within-buffer
        # recompute stored the transformed tree value in _max_priority.
        version = metadata.get("_schema_version", 0)
        self._alpha = metadata["_alpha"]
        self._beta = metadata["_beta"]
        self._eps = metadata["_eps"]
        maxp = tree_map(
            lambda dest, orig: dest.copy_(orig) if dest is not None else orig,
            tuple(self._max_priority),
            tuple(metadata["_max_priority"]),
        )
        if all(x is None for x in self._max_priority):
            self._max_priority = maxp
        _max_capacity = metadata["_max_capacity"]
        if _max_capacity != self._max_capacity:
            raise RuntimeError(
                f"max capacity of loaded metadata ({_max_capacity}) differs from self._max_capacity ({self._max_capacity})."
            )
        mm_st = MemoryMappedTensor.from_filename(
            shape=(self._max_capacity,),
            dtype=torch.float64,
            filename=path / "sumtree.memmap",
        )
        mm_mt = MemoryMappedTensor.from_filename(
            shape=(self._max_capacity,),
            dtype=torch.float64,
            filename=path / "mintree.memmap",
        )
        for i, elt in enumerate(mm_st.tolist()):
            self._sum_tree[i] = elt
        for i, elt in enumerate(mm_mt.tolist()):
            self._min_tree[i] = elt
        self._pending_updates = []
        if (
            version < 1
            and self._max_priority_within_buffer
            and self._max_priority[0] is not None
        ):
            self._recompute_max_priority_from_tree()
