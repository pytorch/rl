"""Recurrent matmul precision control for the triton RNN backend.

The triton GRU/LSTM kernels use ``tl.dot`` for the hidden-to-hidden matmul.
On Ampere/Hopper/Blackwell ``tl.dot`` defaults to TF32 with fp32 inputs,
which yields ~10 bits of mantissa accuracy. The PyTorch wrapper around the
kernel also performs an input-to-hidden ``F.linear`` and two outer-product
weight matmuls through cuBLAS, which by default honors
``torch.backends.cuda.matmul.allow_tf32`` (``False`` since PyTorch 1.12,
hence IEEE FP32).

This precision mismatch is correct *per-step* but accumulates a systematic
bias over long rollouts and is observable as a small but consistent reward
gap in long PPO training runs (`scan` > `cudnn` > current `triton`). This
module exposes one knob that synchronizes the precision across the kernel
``tl.dot`` and the wrapper ``F.linear`` calls.

Resolution order (most specific wins):

1. Explicit ``recurrent_matmul_precision=`` kwarg on
   :class:`~torchrl.modules.LSTMModule` / :class:`~torchrl.modules.GRUModule`.
2. Process-global override set via :func:`set_recurrent_matmul_precision`.
3. Environment variable ``TORCHRL_RNN_PRECISION`` read at import time.
4. Derived from :func:`torch.get_float32_matmul_precision`.

Modes:

* ``"ieee"`` — full IEEE FP32. ``tl.dot`` runs on CUDA cores (no tensor cores),
  ``F.linear`` IEEE. Slowest, highest precision. Use for correctness diffs
  against the ``scan`` backend.
* ``"tf32x3"`` — three-product tensor-core decomposition (~22 bits of
  mantissa). ``tl.dot`` runs on tensor cores at ~1/3 of plain TF32
  throughput; ``F.linear`` stays IEEE (cuBLAS has no tf32x3 mode).
  This is the recommended default for RL recurrent training on Ampere+.
* ``"tf32"`` — plain TF32. ``tl.dot`` and ``F.linear`` both TF32. Matches
  cuDNN's default. Fastest but has a small systematic bias that compounds
  in long-T RL rollouts.
* ``"auto"`` (default for the kwarg) — defer to the global setter / env var /
  ``torch.get_float32_matmul_precision()``.

Mapping from :func:`torch.get_float32_matmul_precision`:

* ``"highest"`` → ``"ieee"``
* ``"high"``    → ``"tf32x3"`` (deliberately not ``"tf32"`` — see module
                  docstring)
* ``"medium"``  → ``"tf32"``
"""
from __future__ import annotations

import contextlib
import os
import typing
from typing import Literal

import torch

__all__ = [
    "RecurrentMatmulPrecision",
    "get_recurrent_matmul_precision",
    "set_recurrent_matmul_precision",
]

RecurrentMatmulPrecision = Literal["ieee", "tf32", "tf32x3"]

_VALID_MODES: frozenset[str] = frozenset({"ieee", "tf32", "tf32x3"})
_VALID_USER_MODES: frozenset[str] = frozenset({"auto", "ieee", "tf32", "tf32x3"})

_TORCH_TO_RNN_PRECISION: dict[str, RecurrentMatmulPrecision] = {
    "highest": "ieee",
    "high": "tf32x3",
    "medium": "tf32",
}

_ENV_VAR = "TORCHRL_RNN_PRECISION"


def _read_env_default() -> RecurrentMatmulPrecision | None:
    raw = os.environ.get(_ENV_VAR)
    if raw is None:
        return None
    raw = raw.strip().lower()
    if raw == "auto" or raw == "":
        return None
    if raw not in _VALID_MODES:
        raise ValueError(
            f"{_ENV_VAR}={raw!r} is not a valid recurrent matmul precision. "
            f"Expected one of {sorted(_VALID_MODES)} or 'auto'."
        )
    return typing.cast(RecurrentMatmulPrecision, raw)


_GLOBAL_OVERRIDE: RecurrentMatmulPrecision | None = _read_env_default()


def set_recurrent_matmul_precision(mode: str | None) -> None:
    """Set the process-global precision for the triton RNN backend.

    Args:
        mode: One of ``"ieee"``, ``"tf32"``, ``"tf32x3"``, ``"auto"`` or
            ``None``. ``"auto"`` and ``None`` clear the override and fall
            back to :func:`torch.get_float32_matmul_precision` (modulated by
            the ``TORCHRL_RNN_PRECISION`` env var if set).

    The setting is read at every triton GRU/LSTM call, so changes take
    effect immediately. Per-module ``recurrent_matmul_precision=`` kwargs
    still override this global value.
    """
    global _GLOBAL_OVERRIDE
    if mode is None or mode == "auto":
        _GLOBAL_OVERRIDE = _read_env_default()
        return
    if mode not in _VALID_MODES:
        raise ValueError(
            f"Invalid recurrent matmul precision {mode!r}. "
            f"Expected one of {sorted(_VALID_USER_MODES)}."
        )
    _GLOBAL_OVERRIDE = typing.cast(RecurrentMatmulPrecision, mode)


def get_recurrent_matmul_precision() -> RecurrentMatmulPrecision:
    """Resolve the currently effective precision (global level).

    Does not see per-module overrides; those are resolved at the call site
    via :func:`_resolve_precision`.
    """
    if _GLOBAL_OVERRIDE is not None:
        return _GLOBAL_OVERRIDE
    return _TORCH_TO_RNN_PRECISION[torch.get_float32_matmul_precision()]


def _resolve_precision(kwarg: str | None) -> RecurrentMatmulPrecision:
    """Resolve the effective precision for a single kernel call.

    Args:
        kwarg: ``"auto"``, ``None`` or one of the explicit modes from
            :data:`RecurrentMatmulPrecision`. ``"auto"`` defers to
            :func:`get_recurrent_matmul_precision`.
    """
    if kwarg is None or kwarg == "auto":
        return get_recurrent_matmul_precision()
    if kwarg not in _VALID_MODES:
        raise ValueError(
            f"Invalid recurrent matmul precision {kwarg!r}. "
            f"Expected one of {sorted(_VALID_USER_MODES)}."
        )
    return typing.cast(RecurrentMatmulPrecision, kwarg)


def _validate_user_precision(value: str) -> None:
    """Raise if ``value`` is not a valid user-facing precision mode.

    Used by ``LSTMModule``/``GRUModule`` constructors so the failure is
    reported at module construction time, not at the first forward call.
    """
    if value not in _VALID_USER_MODES:
        raise ValueError(
            f"Invalid recurrent_matmul_precision={value!r}. "
            f"Expected one of {sorted(_VALID_USER_MODES)}."
        )


@contextlib.contextmanager
def _cublas_matmul_precision_ctx(precision: RecurrentMatmulPrecision):
    """Temporarily align ``cuda.matmul.allow_tf32`` with the chosen mode.

    Wraps the ``F.linear`` calls inside the triton autograd Function so the
    input-to-hidden matmul and the dW outer-products run at a precision
    consistent with the kernel's ``tl.dot``. cuBLAS has no ``tf32x3`` mode;
    for that mode we keep ``allow_tf32=False`` (IEEE FP32 in cuBLAS, ~23
    bits) which is the closest available match to ``tf32x3``'s ~22 bits.
    """
    if precision == "tf32":
        new_setting = True
    else:
        new_setting = False
    prev = torch.backends.cuda.matmul.allow_tf32
    if prev == new_setting:
        yield
        return
    torch.backends.cuda.matmul.allow_tf32 = new_setting
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
