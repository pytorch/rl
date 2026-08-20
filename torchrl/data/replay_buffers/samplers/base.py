# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import warnings
from abc import ABC, ABCMeta, abstractmethod
from copy import copy
from typing import Any

import torch
from torchrl.data.replay_buffers.storages import Storage


# Maps a "with replacement" sampler class to its "without replacement" counterpart.
# Populated at module import time after the relevant classes are defined.
# Consumed by :class:`_SamplerMeta` to dispatch ``Cls(replacement=False, ...)`` calls
# to ``_REPLACEMENT_DISPATCH[Cls](...)``.
_REPLACEMENT_DISPATCH: dict[type, type] = {}


class _SamplerMeta(ABCMeta):
    """Metaclass enabling ``replacement=False`` dispatch on with-replacement samplers.

    When a class registered in :data:`_REPLACEMENT_DISPATCH` (e.g.
    :class:`RandomSampler`, :class:`SliceSampler`) is instantiated with
    ``replacement=False``, the call is dispatched to its without-replacement
    counterpart (:class:`SamplerWithoutReplacement` or
    :class:`SliceSamplerWithoutReplacement`).

    Calls with ``replacement=True`` (the default) behave exactly like a normal
    instantiation: the ``replacement`` kwarg is popped before the constructor
    runs, so existing ``__init__`` signatures don't need to be changed.

    Passing ``replacement=False`` to a sampler that has no without-replacement
    variant raises :class:`TypeError`. Passing ``replacement=False`` to a
    sampler that is itself already a without-replacement variant is allowed
    and treated as a no-op.
    """

    def __call__(cls, *args, **kwargs):
        if "replacement" in kwargs:
            replacement = kwargs.pop("replacement")
            if not replacement:
                alt = _REPLACEMENT_DISPATCH.get(cls)
                if alt is not None:
                    return alt(*args, **kwargs)
                if cls not in _REPLACEMENT_DISPATCH.values():
                    raise TypeError(
                        f"{cls.__name__} has no without-replacement variant; "
                        "cannot be instantiated with replacement=False."
                    )
        return super().__call__(*args, **kwargs)


class Sampler(ABC, metaclass=_SamplerMeta):
    """A generic sampler base class for composable Replay Buffers."""

    # Some samplers - mainly those without replacement -
    # need to keep track of the number of remaining batches
    _remaining_batches = int(torch.iinfo(torch.int64).max)

    # The RNG is set by the replay buffer
    _rng: torch.Generator | None = None

    @abstractmethod
    def sample(self, storage: Storage, batch_size: int) -> tuple[Any, dict]:
        ...

    def add(self, index: int) -> None:
        return

    def extend(self, index: torch.Tensor) -> None:
        return

    def update_priority(
        self,
        index: int | torch.Tensor,
        priority: float | torch.Tensor,
        *,
        storage: Storage | None = None,
    ) -> dict | None:
        warnings.warn(
            f"Calling update_priority() on a sampler {type(self).__name__} that is not prioritized. Make sure this is the indented behavior."
        )
        return

    def mark_update(
        self, index: int | torch.Tensor, *, storage: Storage | None = None
    ) -> None:
        return

    @property
    def default_priority(self) -> float:
        return 1.0

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        ...

    @abstractmethod
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        ...

    @property
    def ran_out(self) -> bool:
        # by default, samplers never run out
        return False

    @abstractmethod
    def _empty(self):
        ...

    @abstractmethod
    def dumps(self, path):
        ...

    @abstractmethod
    def loads(self, path):
        ...

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def __getstate__(self):
        state = copy(self.__dict__)
        state["_rng"] = None
        return state
