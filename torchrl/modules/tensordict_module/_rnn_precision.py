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

User-facing modes:

Concrete:

* ``"ieee"`` — full IEEE FP32. ``tl.dot`` runs on CUDA cores (no tensor cores),
  ``F.linear`` IEEE. Slowest, highest precision. Use for correctness diffs
  against the ``scan`` backend.
* ``"tf32"`` — plain TF32 (~10 bits of mantissa). ``tl.dot`` and ``F.linear``
  both TF32. Matches cuDNN's default. Fastest on Ampere+ but has a small
  systematic bias that compounds in long-T RL rollouts.
* ``"tf32x3"`` — three-product tensor-core decomposition (~22 bits of
  mantissa). ``tl.dot`` runs on tensor cores at ~1/3 of plain TF32
  throughput; ``F.linear`` stays IEEE (cuBLAS has no tf32x3 mode).
  Recommended for RL recurrent training on Ampere+ when you care about
  matching the scan/cuDNN reward curve.

GPU-aware presets (resolve to a concrete mode at kernel-call time using
``torch.cuda.get_device_capability``):

* ``"fast"`` — best speed on the current GPU.
  Ampere/Hopper/Blackwell → ``"tf32"``. Volta/Turing/CPU/AMD → ``"ieee"``.
* ``"high-prec"`` — best precision while still on tensor cores.
  Ampere/Hopper/Blackwell → ``"tf32x3"``. Volta/Turing/CPU/AMD → ``"ieee"``.

Resolution helper:

* ``"auto"`` (default for the module kwarg) — defer to the global setter /
  env var / ``torch.get_float32_matmul_precision()``.

Mapping from :func:`torch.get_float32_matmul_precision`:

* ``"highest"`` → ``"ieee"``
* ``"high"``    → ``"high-prec"`` (GPU-aware, was ``"tf32x3"`` pre-presets)
* ``"medium"``  → ``"fast"`` (GPU-aware, was ``"tf32"`` pre-presets)

So on a V100/T4 box ``torch.set_float32_matmul_precision("high")`` no longer
silently picks a tf32x3 path that can't use tensor cores anyway — it falls
back to ``"ieee"``.
"""
from __future__ import annotations

import functools
import os
import typing
from typing import Literal

import torch

__all__ = [
    "RecurrentMatmulPrecision",
    "RecurrentMatmulPrecisionUserMode",
    "get_recurrent_matmul_precision",
    "set_recurrent_matmul_precision",
]

# Concrete modes the kernel and cuBLAS path can actually run.
RecurrentMatmulPrecision = Literal["ieee", "tf32", "tf32x3"]

# User-facing modes (what the public API accepts).
RecurrentMatmulPrecisionUserMode = Literal[
    "auto", "fast", "high-prec", "ieee", "tf32", "tf32x3"
]

_VALID_CONCRETE_MODES: frozenset[str] = frozenset({"ieee", "tf32", "tf32x3"})
_VALID_PRESET_MODES: frozenset[str] = frozenset({"fast", "high-prec"})
_VALID_USER_MODES: frozenset[str] = (
    frozenset({"auto"}) | _VALID_PRESET_MODES | _VALID_CONCRETE_MODES
)

# Mapping from PyTorch's float32 matmul precision setting. ``"high"`` maps to
# the GPU-aware preset, not directly to ``"tf32x3"``, so pre-Ampere hardware
# falls back to ``"ieee"`` rather than emulating tf32x3 on CUDA cores.
_TORCH_TO_RNN_PRECISION: dict[str, str] = {
    "highest": "ieee",
    "high": "high-prec",
    "medium": "fast",
}

_ENV_VAR = "TORCHRL_RNN_PRECISION"


def _read_env_default() -> str | None:
    raw = os.environ.get(_ENV_VAR)
    if raw is None:
        return None
    raw = raw.strip().lower()
    if raw == "auto" or raw == "":
        return None
    if raw not in (_VALID_CONCRETE_MODES | _VALID_PRESET_MODES):
        raise ValueError(
            f"{_ENV_VAR}={raw!r} is not a valid recurrent matmul precision. "
            f"Expected one of {sorted(_VALID_USER_MODES)}."
        )
    return raw


# Stores the symbolic value (possibly a preset like ``"fast"``). Resolution
# to a concrete mode happens at kernel-call time so per-device changes are
# honored without restarting the process.
_GLOBAL_OVERRIDE: str | None = _read_env_default()


@functools.lru_cache(maxsize=8)
def _is_tensor_core_capable(device_index: int) -> bool:
    """Whether the device at ``device_index`` has TF32 tensor cores.

    Returns ``True`` for compute capability >= 8.0 (Ampere and newer) on
    NVIDIA GPUs. ROCm/HIP devices return ``False`` because Triton's
    ``tl.dot(..., input_precision="tf32")`` is a no-op there.
    """
    if not torch.cuda.is_available():
        return False
    if getattr(torch.version, "hip", None) is not None:
        return False
    try:
        major, _ = torch.cuda.get_device_capability(device_index)
    except Exception:
        return False
    return major >= 8


def _current_device_index() -> int:
    """Best-effort current CUDA device index for preset resolution."""
    if not torch.cuda.is_available():
        return -1
    try:
        return torch.cuda.current_device()
    except Exception:
        return 0


def _resolve_gpu_preset(preset: str) -> RecurrentMatmulPrecision:
    """Resolve ``"fast"`` / ``"high-prec"`` to a concrete mode for this GPU."""
    if not _is_tensor_core_capable(_current_device_index()):
        return "ieee"
    if preset == "fast":
        return "tf32"
    if preset == "high-prec":
        return "tf32x3"
    raise ValueError(f"Unknown precision preset {preset!r}")


def set_recurrent_matmul_precision(mode: str | None) -> None:
    """Set the process-global precision for the triton RNN backend.

    Args:
        mode: One of ``"ieee"``, ``"tf32"``, ``"tf32x3"``, ``"fast"``,
            ``"high-prec"``, ``"auto"`` or ``None``. ``"auto"`` and ``None``
            clear the override and fall back to
            :func:`torch.get_float32_matmul_precision` (modulated by the
            ``TORCHRL_RNN_PRECISION`` env var if set). ``"fast"`` and
            ``"high-prec"`` are stored symbolically and resolve to a concrete
            mode at every kernel call based on the active CUDA device.

    The setting is read at every triton GRU/LSTM call, so changes take
    effect immediately. Per-module ``recurrent_matmul_precision=`` kwargs
    still override this global value.
    """
    global _GLOBAL_OVERRIDE
    if mode is None or mode == "auto":
        _GLOBAL_OVERRIDE = _read_env_default()
        return
    if mode not in (_VALID_CONCRETE_MODES | _VALID_PRESET_MODES):
        raise ValueError(
            f"Invalid recurrent matmul precision {mode!r}. "
            f"Expected one of {sorted(_VALID_USER_MODES)}."
        )
    _GLOBAL_OVERRIDE = mode


def get_recurrent_matmul_precision() -> RecurrentMatmulPrecision:
    """Resolve the currently effective precision to a concrete mode.

    Always returns one of ``"ieee"``, ``"tf32"`` or ``"tf32x3"``. The result
    is what the kernel actually runs at, including preset / GPU resolution.
    Does not see per-module overrides; those are resolved at the call site
    via :func:`_resolve_precision`.
    """
    return _resolve_precision(None)


def _resolve_precision(kwarg: str | None) -> RecurrentMatmulPrecision:
    """Resolve the effective precision for a single kernel call.

    Args:
        kwarg: ``"auto"``, ``None`` or one of the public modes. ``"auto"``
            defers to the global override / env var /
            :func:`torch.get_float32_matmul_precision`. Presets ``"fast"`` /
            ``"high-prec"`` resolve to a concrete mode for the current GPU.
    """
    if kwarg is None or kwarg == "auto":
        # Walk the resolution chain: global override (already populated from
        # the env var at import time) → torch precision setting.
        if _GLOBAL_OVERRIDE is not None:
            return _resolve_precision(_GLOBAL_OVERRIDE)
        torch_mode = torch.get_float32_matmul_precision()
        return _resolve_precision(_TORCH_TO_RNN_PRECISION[torch_mode])
    if kwarg in _VALID_PRESET_MODES:
        return _resolve_gpu_preset(kwarg)
    if kwarg in _VALID_CONCRETE_MODES:
        return typing.cast(RecurrentMatmulPrecision, kwarg)
    raise ValueError(
        f"Invalid recurrent matmul precision {kwarg!r}. "
        f"Expected one of {sorted(_VALID_USER_MODES)}."
    )


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


def _cublas_wants_tf32(precision: RecurrentMatmulPrecision) -> bool:
    """Whether ``cuda.matmul.allow_tf32`` should be ``True`` for this mode.

    cuBLAS has no ``tf32x3`` mode; for ``tf32x3`` we want IEEE FP32 in
    cuBLAS (~23 bits of mantissa) as the closest match to the kernel's
    ~22-bit ``tf32x3`` accumulation. Only plain ``tf32`` wants cuBLAS TF32.
    """
    return precision == "tf32"


def _maybe_enable_tf32(precision: RecurrentMatmulPrecision) -> bool | None:
    """Flip ``cuda.matmul.allow_tf32`` to match ``precision`` if needed.

    Returns the previous value when the global flag was actually mutated, or
    ``None`` when no mutation happened. Pair every non-``None`` return with
    :func:`_restore_tf32` to put the flag back.

    Implemented as a plain pair of helpers (rather than a
    ``@contextmanager`` generator) because the autograd Function's forward
    and backward run inside ``torch.compile`` regions, and compiled_autograd
    handles a couple of conditional assignments much better than the
    generator's yield/cleanup machinery.
    """
    want = _cublas_wants_tf32(precision)
    prev = torch.backends.cuda.matmul.allow_tf32
    if prev == want:
        return None
    torch.backends.cuda.matmul.allow_tf32 = want
    return prev


def _restore_tf32(prev: bool | None) -> None:
    """Restore ``cuda.matmul.allow_tf32`` if :func:`_maybe_enable_tf32` flipped it."""
    if prev is None:
        return
    torch.backends.cuda.matmul.allow_tf32 = prev
