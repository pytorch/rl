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
    "PostTrainingLossOutputProtocol",
    "assert_satisfies_protocol",
]


@runtime_checkable
class PostTrainingBufferProtocol(Protocol):
    """Minimal interface a replay buffer must expose for post-training loops.

    **TensorDict key contract** — samples must carry:

    * ``("next", "reward")`` — reward signal, shape ``[B, ...]``.
    * ``("tokens", "full")`` — full token sequence, shape ``[B, T]``.
    * ``("tokens", "response")`` — response tokens, shape ``[B, R]`` or ragged.
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


class PostTrainingLossOutputProtocol(Protocol):
    """Minimal interface a loss-output object must expose.

    ``GRPOLossOutput`` and ``SFTLossOutput`` satisfy this protocol structurally.

    .. note::
        Both are ``TensorClass`` subclasses so standard ``isinstance`` checks
        will not work. Use :func:`assert_satisfies_protocol` instead.

    **Optional fields** read via ``getattr`` by ``PostTrainingLogger``:
    ``loss_objective``, ``loss_sft``, ``clip_fraction``, ``kl_approx``,
    ``ESS``, ``entropy``, ``loss_entropy``, ``loss_kl_to_ref``, ``kl_to_ref``,
    ``loss_kl_to_inference``, ``kl_to_inference``.
    """

    @property
    def loss_objective(self) -> torch.Tensor | None:
        """Primary policy loss (GRPO/PPO), or ``None`` if not computed."""
        ...

    @property
    def loss_sft(self) -> torch.Tensor | None:
        """SFT loss (SFT / Expert Iteration outputs)."""
        ...


# Primary loss fields checked by assert_satisfies_protocol for TensorClass outputs.
_LOSS_OUTPUT_FIELDS: tuple[str, ...] = ("loss_objective", "loss_sft")


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
    if protocol is PostTrainingLossOutputProtocol:
        # GRPOLossOutput / SFTLossOutput are TensorClass subclasses: their fields
        # live in TensorDict internals, not Python __dict__, so isinstance fails.
        has_any = any(getattr(obj, f, None) is not None for f in _LOSS_OUTPUT_FIELDS)
        if not has_any:
            label = f'"{name}" ' if name else ""
            raise TypeError(
                f"Object {label}of type '{type(obj).__name__}' does not satisfy "
                f"'PostTrainingLossOutputProtocol'.\n"
                f"  Must expose at least one of: {', '.join(_LOSS_OUTPUT_FIELDS)}"
            )
        return

    if not isinstance(obj, protocol):
        label = f'"{name}" ' if name else ""
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
