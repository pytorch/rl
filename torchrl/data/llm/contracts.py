# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Stable component boundaries for LLM post-training external-loop interop (RFC #3948, WS1).

Defines ``typing.Protocol`` interfaces for replay buffers, collectors, and
loss outputs so external training loops (TRL, NeMo-RL, custom) can consume
individual TorchRL components without depending on concrete classes.
"""
from __future__ import annotations

from typing import Any, Iterator, runtime_checkable

import torch
from tensordict import TensorDictBase
from typing_extensions import Protocol

__all__ = [
    "PostTrainingBufferProtocol",
    "PostTrainingCollectorProtocol",
    "GRPOLossOutputProtocol",
    "SFTLossOutputProtocol",
    "assert_satisfies_protocol",
]


@runtime_checkable
class PostTrainingBufferProtocol(Protocol):
    """Minimal interface a replay buffer must expose for post-training loops.

    **TensorDict key contract** — samples must carry:

    * ``("next", "reward")`` — reward signal, shape ``[B, ...]``.
    * ``("tokens", "full")`` — full token sequence, shape ``[B, T]``.
    * ``("tokens", "response")`` — response tokens, shape ``[B, R]`` or ragged.

    Example::

        >>> from torchrl.data import ReplayBuffer, LazyTensorStorage
        >>> from torchrl.data.llm.contracts import PostTrainingBufferProtocol
        >>> rb = ReplayBuffer(storage=LazyTensorStorage(100))
        >>> isinstance(rb, PostTrainingBufferProtocol)
        True
    """

    def extend(self, data: TensorDictBase) -> None:
        """Append a batch of trajectories to the buffer."""
        ...

    def sample(self, batch_size: int | None = None) -> TensorDictBase:
        """Sample a batch from the buffer."""
        ...

    @property
    def write_count(self) -> int:
        """Total number of samples written since creation."""
        ...


@runtime_checkable
class PostTrainingCollectorProtocol(Protocol):
    """Minimal interface an LLM rollout collector must expose.

    **TensorDict key contract** — each yielded batch must carry:

    * ``("tokens", "full")`` — full token sequence, shape ``[B, T]``.
    * ``("tokens", "prompt")`` — prompt tokens, shape ``[B, P]``.
    * ``("tokens", "response")`` — response tokens, shape ``[B, R]``.
    * ``("next", "reward")`` — reward signal, shape ``[B, ...]``.
    * ``("masks", "all_attention_mask")`` — boolean mask, shape ``[B, T]``.

    Example::

        >>> from torchrl.data.llm.contracts import PostTrainingCollectorProtocol
        >>> class MyCollector:
        ...     def update_policy_weights_(self, policy_weights=None, *, worker_ids=None): ...
        ...     def __iter__(self): return iter([])
        ...     def __next__(self): raise StopIteration
        >>> isinstance(MyCollector(), PostTrainingCollectorProtocol)
        True
    """

    def update_policy_weights_(
        self,
        policy_weights: Any | None = None,
        *,
        worker_ids: list[int] | None = None,
    ) -> None:
        """Push updated policy weights to inference workers."""
        ...

    def __iter__(self) -> Iterator[TensorDictBase]:
        ...

    def __next__(self) -> TensorDictBase:
        ...


class GRPOLossOutputProtocol(Protocol):
    """Interface satisfied by :class:`~torchrl.objectives.llm.GRPOLossOutput`.

    .. note::
        ``GRPOLossOutput`` is a ``TensorClass`` subclass whose fields live in
        TensorDict internals rather than Python ``__dict__``.  Use
        :func:`assert_satisfies_protocol` instead of ``isinstance``.

    **Field contract** — fields read by ``PostTrainingLogger`` via ``getattr``:
    ``loss_objective``, ``clip_fraction``, ``kl_approx``, ``ESS``,
    ``entropy``, ``loss_entropy``, ``loss_kl_to_ref``, ``kl_to_ref``,
    ``loss_kl_to_inference``, ``kl_to_inference``.

    Example::

        >>> import torch
        >>> from torchrl.objectives.llm.grpo import GRPOLossOutput
        >>> from torchrl.data.llm.contracts import GRPOLossOutputProtocol, assert_satisfies_protocol
        >>> out = GRPOLossOutput(
        ...     loss_objective=torch.tensor(0.5),
        ...     clip_fraction=torch.tensor(0.1),
        ...     kl_approx=torch.tensor(0.01),
        ...     ESS=torch.tensor(32.0),
        ... )
        >>> assert_satisfies_protocol(out, GRPOLossOutputProtocol)
    """

    @property
    def loss_objective(self) -> torch.Tensor:
        """Primary GRPO policy loss."""
        ...


class SFTLossOutputProtocol(Protocol):
    """Interface satisfied by :class:`~torchrl.objectives.llm.SFTLossOutput`.

    .. note::
        ``SFTLossOutput`` is a ``TensorClass`` subclass whose fields live in
        TensorDict internals rather than Python ``__dict__``.  Use
        :func:`assert_satisfies_protocol` instead of ``isinstance``.

    **Field contract** — fields read by ``PostTrainingLogger`` via ``getattr``:
    ``loss_sft``, ``kl_to_ref``, ``loss_kl_to_ref``.

    Example::

        >>> import torch
        >>> from torchrl.objectives.llm.sft import SFTLossOutput
        >>> from torchrl.data.llm.contracts import SFTLossOutputProtocol, assert_satisfies_protocol
        >>> out = SFTLossOutput(loss_sft=torch.tensor(0.3))
        >>> assert_satisfies_protocol(out, SFTLossOutputProtocol)
    """

    @property
    def loss_sft(self) -> torch.Tensor:
        """Supervised fine-tuning loss."""
        ...


# Maps each loss-output protocol to the single field it requires.
_LOSS_PROTOCOL_FIELDS: dict[type, str] = {
    GRPOLossOutputProtocol: "loss_objective",
    SFTLossOutputProtocol: "loss_sft",
}


def assert_satisfies_protocol(
    obj: Any,
    protocol: type,
    *,
    name: str = "",
) -> None:
    """Raise :class:`TypeError` if *obj* does not satisfy *protocol*.

    Development utility — call once at the start of a training loop to
    catch integration bugs early. Not intended for hot paths.

    Args:
        obj: Object to validate.
        protocol: A :class:`~typing.Protocol` to check against.
        name: Optional label included in the error message.

    Raises:
        TypeError: If *obj* does not satisfy *protocol*.

    Example::

        >>> from torchrl.data import ReplayBuffer, LazyTensorStorage
        >>> from torchrl.data.llm.contracts import (
        ...     PostTrainingBufferProtocol, assert_satisfies_protocol,
        ... )
        >>> rb = ReplayBuffer(storage=LazyTensorStorage(100))
        >>> assert_satisfies_protocol(rb, PostTrainingBufferProtocol, name="rb")
    """
    label = f'"{name}" ' if name else ""

    if protocol in _LOSS_PROTOCOL_FIELDS:
        # GRPOLossOutput / SFTLossOutput are TensorClass subclasses: their fields
        # live in TensorDict internals, not Python __dict__, so isinstance fails.
        required_field = _LOSS_PROTOCOL_FIELDS[protocol]
        if getattr(obj, required_field, None) is None:
            raise TypeError(
                f"Object {label}of type '{type(obj).__name__}' does not satisfy "
                f"'{protocol.__name__}'.\n"
                f"  Must expose a non-None '{required_field}' field."
            )
        return

    if not isinstance(obj, protocol):
        protocol_name = getattr(protocol, "__name__", str(protocol))
        missing = [
            attr
            for attr in getattr(protocol, "__protocol_attrs__", [])
            if not hasattr(obj, attr)
        ]
        missing_str = f"  Missing: {', '.join(missing)}" if missing else ""
        raise TypeError(
            f"Object {label}of type '{type(obj).__name__}' does not satisfy "
            f"'{protocol_name}'.\n{missing_str}"
        )
