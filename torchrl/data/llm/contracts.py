# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Defines validation helper functions for replay buffers, collectors, and
loss outputs so external training loops (TRL, NeMo-RL, custom) can consume
individual TorchRL components with confidence.
"""
from __future__ import annotations

from typing import Any

__all__ = [
    "assert_buffer_contract",
    "assert_collector_contract",
    "assert_loss_contract",
]


def assert_buffer_contract(buffer: Any) -> None:
    """Assert that a replay buffer exposes the necessary methods for post-training.

    **TensorDict key contract** — samples must carry:

    * ``("next", "reward")`` — reward signal, shape ``[B, ...]``.
    * ``("tokens", "full")`` — full token sequence, shape ``[B, T]``.
    * ``("tokens", "response")`` — response tokens, shape ``[B, R]`` or ragged.

    Args:
        buffer: The buffer object to validate.

    Raises:
        TypeError: If the buffer is missing ``extend``, ``sample``, or ``write_count``.
    """
    missing = [
        attr
        for attr in ("extend", "sample", "write_count")
        if not hasattr(buffer, attr)
    ]
    if missing:
        raise TypeError(
            f"Object of type '{type(buffer).__name__}' does not satisfy the "
            f"post-training buffer contract.\n  Missing: {', '.join(missing)}"
        )


def assert_collector_contract(collector: Any) -> None:
    """Assert that a rollout collector exposes the necessary methods.

    **TensorDict key contract** — each yielded batch must carry:

    * ``("tokens", "full")`` — full token sequence, shape ``[B, T]``.
    * ``("tokens", "prompt")`` — prompt tokens, shape ``[B, P]``.
    * ``("tokens", "response")`` — response tokens, shape ``[B, R]``.
    * ``("next", "reward")`` — reward signal, shape ``[B, ...]``.
    * ``("masks", "all_attention_mask")`` — boolean mask, shape ``[B, T]``.

    Args:
        collector: The collector object to validate.

    Raises:
        TypeError: If the collector is missing ``update_policy_weights_``, ``__iter__``, or ``__next__``.
    """
    missing = [
        attr
        for attr in ("update_policy_weights_", "__iter__", "__next__")
        if not hasattr(collector, attr)
    ]
    if missing:
        raise TypeError(
            f"Object of type '{type(collector).__name__}' does not satisfy the "
            f"post-training collector contract.\n  Missing: {', '.join(missing)}"
        )


def assert_loss_contract(loss_output: Any, *, loss_type: str = "grpo") -> None:
    """Assert that a loss-output object exposes the necessary fields.

    **Field contract** — fields read by ``PostTrainingLogger`` via ``getattr``:

    * For GRPO: ``loss_objective``, ``clip_fraction``, ``kl_approx``, ``ESS``,
      ``entropy``, ``loss_entropy``, ``loss_kl_to_ref``, ``kl_to_ref``,
      ``loss_kl_to_inference``, ``kl_to_inference``.
    * For SFT: ``loss_sft``, ``kl_to_ref``, ``loss_kl_to_ref``.

    Args:
        loss_output: The loss output object (usually a ``TensorClass``) to validate.
        loss_type: The expected loss type, either ``"grpo"`` or ``"sft"``.

    Raises:
        TypeError: If the required field (``loss_objective`` or ``loss_sft``) is missing or None.
        ValueError: If an unknown ``loss_type`` is requested.
    """
    if loss_type == "grpo":
        required_field = "loss_objective"
    elif loss_type == "sft":
        required_field = "loss_sft"
    else:
        raise ValueError(f"Unknown loss_type '{loss_type}'. Expected 'grpo' or 'sft'.")

    if getattr(loss_output, required_field, None) is None:
        raise TypeError(
            f"Object of type '{type(loss_output).__name__}' does not satisfy the "
            f"'{loss_type}' loss contract.\n"
            f"  Must expose a non-None '{required_field}' field."
        )
