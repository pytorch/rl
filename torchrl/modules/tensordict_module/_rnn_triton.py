# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Triton kernels for GRU / LSTM forward and backward with intermediate resets.

These are the building blocks behind the ``recurrent_backend="triton"`` option
on :class:`~torchrl.modules.GRUModule` and :class:`~torchrl.modules.LSTMModule`.

For smaller hidden sizes, the kernels fuse the whole time loop into one CUDA
launch and apply the ``is_init`` reset mask cheaply inside the loop. Large
hidden sizes use one tiled launch per time step to keep each Triton program
small enough to compile. These paths are especially beneficial for RL training,
where resets can occur at any time step inside a rollout and the cuDNN
``pack_padded_sequence`` / ``pad_packed_sequence`` round trip becomes the
bottleneck.

Both forward and backward kernels are K-tiled along the recurrent contraction
axis so a single ``[H, H]`` weight slab fits in shared memory at any hidden
size we care about in RL.

Limitations of this prototype:

* Low-level kernels operate on one layer and one direction at a time.
  Multilayer module wrappers therefore autotune once per layer shape on the
  first call.
* No projection (``proj_size``) in the low-level kernel.
* ``hidden_size`` is internally padded to the next power of two (kept on the
  Python side; no in-kernel masking).
* The autograd wrapper saves per-layer gate activations explicitly. Multilayer
  execution scales this activation memory linearly with the number of layers,
  unlike cuDNN's opaque ``reserve_space``.
* ``compute_dtype`` controls the matmul precision: ``torch.float32`` (default,
  TF32 on Ampere/Hopper, matching ``torch.nn.GRU`` / ``LSTM`` behavior) or
  ``torch.bfloat16`` (twice the SMEM headroom, ~7-bit mantissa).
"""
from __future__ import annotations

import importlib.metadata
import os

import torch
import torch.nn.functional as F
from packaging import version

from torchrl.modules.tensordict_module._rnn_precision import (
    _maybe_enable_tf32,
    _resolve_precision,
    _restore_tf32,
    RecurrentMatmulPrecisionUserMode,
)


def _check_triton_available() -> bool:
    """True if the installed Triton exposes everything this module needs.

    The backend's kernels rely on Triton's libdevice ``tanh`` and on a backward
    path that uses ``tl.atomic_add`` with a 2-D mask, which older Triton
    compilers reject. The version is read from package metadata rather than
    probing libdevice via ``find_spec`` because older Triton builds lack the
    ``triton.language.extra`` parent and that probe would raise
    ``ModuleNotFoundError`` at torchrl import time. Older
    installations fall back transparently to the ``scan`` / ``pad`` backends.
    """
    try:
        triton_version = importlib.metadata.version("triton")
    except importlib.metadata.PackageNotFoundError:
        return False
    return version.parse(triton_version) >= version.parse("2.2")


_has_triton = _check_triton_available()

if _has_triton:
    import triton
    import triton.language as tl

    # BLOCK_K=16 is the smallest, dictating the minimum padded hidden size.
    _MIN_H_PAD = 16

    _FWD_CONFIGS = [
        triton.Config({"BLOCK_B": 4, "BLOCK_K": 16}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_B": 4, "BLOCK_K": 32}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_B": 4, "BLOCK_K": 64}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_B": 8, "BLOCK_K": 16}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_B": 8, "BLOCK_K": 32}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_B": 8, "BLOCK_K": 64}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_B": 16, "BLOCK_K": 16}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_B": 16, "BLOCK_K": 32}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_B": 16, "BLOCK_K": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_B": 32, "BLOCK_K": 16}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_B": 32, "BLOCK_K": 32}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_B": 32, "BLOCK_K": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_B": 32, "BLOCK_K": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_B": 32, "BLOCK_K": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_B": 64, "BLOCK_K": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_B": 64, "BLOCK_K": 64}, num_warps=8, num_stages=1),
    ]

    _BWD_CONFIGS = [
        triton.Config({"BLOCK_B": 4, "BLOCK_K": 16}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_B": 4, "BLOCK_K": 32}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_B": 4, "BLOCK_K": 64}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_B": 8, "BLOCK_K": 16}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_B": 8, "BLOCK_K": 32}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_B": 8, "BLOCK_K": 64}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_B": 16, "BLOCK_K": 16}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_B": 16, "BLOCK_K": 32}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_B": 16, "BLOCK_K": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_B": 32, "BLOCK_K": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_B": 32, "BLOCK_K": 64}, num_warps=4, num_stages=1),
    ]

    # Some older Triton versions read ``prune_configs_by`` as a plain dict and
    # require all three keys to be present. Newer ones use ``.get`` and treat
    # ``None`` as "no perf model / no top-k". Spelling them out keeps the
    # autotune wiring backwards-compatible.
    _PRUNE_BY_TEMPLATE = {
        "perf_model": None,
        "top_k": None,
    }

    def _prune_large_hidden_configs(configs, named_args, **kwargs):
        """Drop configs that are invalid or too large for the padded H."""
        H = kwargs.get("H") or named_args.get("H")
        if H is None:
            return configs
        min_block_b = 16
        max_block_b = None
        max_block_k = None
        if H >= 256:
            min_block_b = None
            max_block_b = 8
            max_block_k = 64
        out = [
            c
            for c in configs
            if c.kwargs["BLOCK_K"] <= H and H % c.kwargs["BLOCK_K"] == 0
            and (min_block_b is None or c.kwargs["BLOCK_B"] >= min_block_b)
            and (max_block_b is None or c.kwargs["BLOCK_B"] <= max_block_b)
            and (max_block_k is None or c.kwargs["BLOCK_K"] <= max_block_k)
        ]
        return out or [min(configs, key=lambda c: c.kwargs["BLOCK_K"])]

    def _env_int(name: str, default: int) -> int:
        """Read a positive int override from the environment, else ``default``."""
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            value = int(raw)
        except ValueError:
            return default
        return value if value > 0 else default

    # Hidden tiling needs a global synchronization between recurrent time steps
    # because h[t + 1] depends on every hidden tile of h[t], so the tiled path
    # issues one launch per step -- correct at any hidden size but launch-bound
    # (measured ~3-5x slower than the fused path at H=256, B=8000 on H200).
    # The fused single-launch path keeps the recurrent state in registers across
    # the whole sequence and is much faster, but a full-hidden program keeps
    # ``n_gates`` ``[BLOCK_B, H]`` fp32 accumulators live: the binding limit is
    # registers *per thread* (255 on every NVIDIA arch since Kepler), which the
    # autotuner already hits at H=256 for LSTM (it falls to BLOCK_B=4 /
    # num_warps=1). At H=512 fused spills and ptxas compile time explodes, so
    # tiling takes over there. Because that ceiling is set by the 255-reg limit
    # -- not by SMEM or register-file size, which is what device properties
    # expose and which barely varies across H100/A100/V100 -- the crossover is
    # effectively arch-invariant and is a constant rather than device-derived.
    # All knobs stay env-overridable as a per-arch escape hatch (BLOCK_B=16 is
    # the bare tensor-core minimum and is usually far too small for large batch).
    _FWD_TILED_H_MIN = _env_int("TORCHRL_RNN_TRITON_TILED_H_MIN", 512)
    _FWD_TILED_BLOCK_B = _env_int("TORCHRL_RNN_TRITON_TILED_BLOCK_B", 16)
    _FWD_TILED_BLOCK_H = _env_int("TORCHRL_RNN_TRITON_TILED_BLOCK_H", 64)
    _FWD_TILED_BLOCK_K = _env_int("TORCHRL_RNN_TRITON_TILED_BLOCK_K", 64)
    _FWD_TILED_NUM_WARPS = _env_int("TORCHRL_RNN_TRITON_TILED_NUM_WARPS", 4)

    # ------------------------------------------------------------------------
    # GRU forward
    # ------------------------------------------------------------------------

    @triton.autotune(
        configs=_FWD_CONFIGS,
        key=["T", "H", "INPUT_PRECISION", "SAVE_GATES"],
        prune_configs_by={
            **_PRUNE_BY_TEMPLATE,
            "early_config_prune": _prune_large_hidden_configs,
        },
    )
    @triton.jit
    def _gru_fwd_kernel(
        gates_x_ptr,
        hidden_ptr,  # [B, T, H] per-step reset values; hidden[:, 0] is the initial state
        w_r_ptr,
        w_z_ptr,
        w_n_ptr,
        b_hh_ptr,
        is_init_ptr,
        out_ptr,
        save_r_ptr,
        save_z_ptr,
        save_n_ptr,
        save_gh_n_ptr,
        h_final_ptr,
        B,
        T,
        H: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_K: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        SAVE_GATES: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b_off = pid * BLOCK_B + tl.arange(0, BLOCK_B)
        b_off_i64 = b_off.to(tl.int64)
        h_off = tl.arange(0, H)
        k_inner = tl.arange(0, BLOCK_K)
        mask_b = b_off < B
        N_K: tl.constexpr = H // BLOCK_K

        # Initial state = hidden[:, 0, :].
        h = tl.load(
            hidden_ptr + b_off_i64[:, None] * (T * H) + 0 * H + h_off[None, :],
            mask=mask_b[:, None],
            other=0.0,
        )
        b_r = tl.load(b_hh_ptr + 0 * H + h_off)
        b_z = tl.load(b_hh_ptr + 1 * H + h_off)
        b_n = tl.load(b_hh_ptr + 2 * H + h_off)

        for t in range(T):
            base_x = b_off_i64[:, None] * (T * 3 * H) + t * (3 * H)
            gx_r = tl.load(
                gates_x_ptr + base_x + 0 * H + h_off[None, :],
                mask=mask_b[:, None],
                other=0.0,
            )
            gx_z = tl.load(
                gates_x_ptr + base_x + 1 * H + h_off[None, :],
                mask=mask_b[:, None],
                other=0.0,
            )
            gx_n = tl.load(
                gates_x_ptr + base_x + 2 * H + h_off[None, :],
                mask=mask_b[:, None],
                other=0.0,
            )

            # ``is_init`` is a ``torch.bool`` tensor; ``tl.load`` returns it as
            # ``uint8`` (the underlying storage dtype). Recent Triton releases
            # warn / error out on ``tl.where`` with non-i1 conditions, so
            # convert explicitly here once and reuse the i1 value below.
            is_init = (
                tl.load(is_init_ptr + b_off_i64 * T + t, mask=mask_b, other=False) != 0
            )
            reset_h = tl.load(
                hidden_ptr + b_off_i64[:, None] * (T * H) + t * H + h_off[None, :],
                mask=mask_b[:, None],
                other=0.0,
            )
            h = tl.where(is_init[:, None], reset_h, h)

            gh_r = tl.zeros([BLOCK_B, H], dtype=tl.float32)
            gh_z = tl.zeros([BLOCK_B, H], dtype=tl.float32)
            gh_n = tl.zeros([BLOCK_B, H], dtype=tl.float32)

            for k_iter in tl.static_range(N_K):
                k_off = k_iter * BLOCK_K + k_inner
                if t == 0:
                    h_chunk = tl.load(
                        hidden_ptr + b_off_i64[:, None] * (T * H) + 0 * H + k_off[None, :],
                        mask=mask_b[:, None],
                        other=0.0,
                    )
                else:
                    h_prev_stored = tl.load(
                        out_ptr
                        + b_off_i64[:, None] * (T * H)
                        + (t - 1) * H
                        + k_off[None, :],
                        mask=mask_b[:, None],
                        other=0.0,
                    )
                    reset_chunk = tl.load(
                        hidden_ptr + b_off_i64[:, None] * (T * H) + t * H + k_off[None, :],
                        mask=mask_b[:, None],
                        other=0.0,
                    )
                    h_chunk = tl.where(is_init[:, None], reset_chunk, h_prev_stored)

                w_r_chunk = tl.load(w_r_ptr + k_off[:, None] * H + h_off[None, :])
                h_chunk_w = h_chunk.to(w_r_chunk.dtype)
                gh_r += tl.dot(h_chunk_w, w_r_chunk, input_precision=INPUT_PRECISION)
                w_z_chunk = tl.load(w_z_ptr + k_off[:, None] * H + h_off[None, :])
                gh_z += tl.dot(h_chunk_w, w_z_chunk, input_precision=INPUT_PRECISION)
                w_n_chunk = tl.load(w_n_ptr + k_off[:, None] * H + h_off[None, :])
                gh_n += tl.dot(h_chunk_w, w_n_chunk, input_precision=INPUT_PRECISION)

            gh_r += b_r[None, :]
            gh_z += b_z[None, :]
            gh_n += b_n[None, :]

            r = tl.sigmoid(gx_r + gh_r)
            z = tl.sigmoid(gx_z + gh_z)
            n = tl.extra.cuda.libdevice.tanh(gx_n + r * gh_n)
            h = n + z * (h - n)

            base_out = b_off_i64[:, None] * (T * H) + t * H + h_off[None, :]
            tl.store(out_ptr + base_out, h, mask=mask_b[:, None])
            if SAVE_GATES:
                tl.store(save_r_ptr + base_out, r, mask=mask_b[:, None])
                tl.store(save_z_ptr + base_out, z, mask=mask_b[:, None])
                tl.store(save_n_ptr + base_out, n, mask=mask_b[:, None])
                tl.store(save_gh_n_ptr + base_out, gh_n, mask=mask_b[:, None])

        tl.store(
            h_final_ptr + b_off_i64[:, None] * H + h_off[None, :], h, mask=mask_b[:, None]
        )

    @triton.jit(do_not_specialize=["t"])
    def _gru_fwd_step_tiled_h_kernel(
        gates_x_ptr,
        hidden_ptr,  # [B, T, H] per-step reset values; hidden[:, 0] is the initial state
        h_prev_ptr,  # [B, H] previous hidden state for this step
        w_r_ptr,
        w_z_ptr,
        w_n_ptr,
        b_hh_ptr,
        is_init_ptr,
        out_ptr,
        save_r_ptr,
        save_z_ptr,
        save_n_ptr,
        save_gh_n_ptr,
        h_next_ptr,  # [B, H] next hidden state for this step
        B,
        T,
        H: tl.constexpr,
        t,
        BLOCK_B: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_K: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        SAVE_GATES: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        b_off = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        b_off_i64 = b_off.to(tl.int64)
        h_off = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        k_inner = tl.arange(0, BLOCK_K)
        mask_b = b_off < B
        mask_h = h_off < H
        mask_bh = mask_b[:, None] & mask_h[None, :]
        N_K: tl.constexpr = H // BLOCK_K

        is_init = tl.load(is_init_ptr + b_off_i64 * T + t, mask=mask_b, other=False) != 0
        reset_h = tl.load(
            hidden_ptr + b_off_i64[:, None] * (T * H) + t * H + h_off[None, :],
            mask=mask_bh,
            other=0.0,
        )
        h_prev = tl.load(
            h_prev_ptr + b_off_i64[:, None] * H + h_off[None, :],
            mask=mask_bh,
            other=0.0,
        )
        h_prev = tl.where(is_init[:, None], reset_h, h_prev)

        base_x = b_off_i64[:, None] * (T * 3 * H) + t * (3 * H)
        gx_r = tl.load(
            gates_x_ptr + base_x + 0 * H + h_off[None, :],
            mask=mask_bh,
            other=0.0,
        )
        gx_z = tl.load(
            gates_x_ptr + base_x + 1 * H + h_off[None, :],
            mask=mask_bh,
            other=0.0,
        )
        gx_n = tl.load(
            gates_x_ptr + base_x + 2 * H + h_off[None, :],
            mask=mask_bh,
            other=0.0,
        )

        gh_r = tl.zeros([BLOCK_B, BLOCK_H], dtype=tl.float32)
        gh_z = tl.zeros([BLOCK_B, BLOCK_H], dtype=tl.float32)
        gh_n = tl.zeros([BLOCK_B, BLOCK_H], dtype=tl.float32)

        for k_iter in tl.static_range(N_K):
            k_off = k_iter * BLOCK_K + k_inner
            h_chunk = tl.load(
                h_prev_ptr + b_off_i64[:, None] * H + k_off[None, :],
                mask=mask_b[:, None],
                other=0.0,
            )
            reset_chunk = tl.load(
                hidden_ptr + b_off_i64[:, None] * (T * H) + t * H + k_off[None, :],
                mask=mask_b[:, None],
                other=0.0,
            )
            h_chunk = tl.where(is_init[:, None], reset_chunk, h_chunk)

            w_r_chunk = tl.load(
                w_r_ptr + k_off[:, None] * H + h_off[None, :],
                mask=mask_h[None, :],
                other=0.0,
            )
            h_chunk_w = h_chunk.to(w_r_chunk.dtype)
            gh_r += tl.dot(h_chunk_w, w_r_chunk, input_precision=INPUT_PRECISION)
            w_z_chunk = tl.load(
                w_z_ptr + k_off[:, None] * H + h_off[None, :],
                mask=mask_h[None, :],
                other=0.0,
            )
            gh_z += tl.dot(h_chunk_w, w_z_chunk, input_precision=INPUT_PRECISION)
            w_n_chunk = tl.load(
                w_n_ptr + k_off[:, None] * H + h_off[None, :],
                mask=mask_h[None, :],
                other=0.0,
            )
            gh_n += tl.dot(h_chunk_w, w_n_chunk, input_precision=INPUT_PRECISION)

        b_r = tl.load(b_hh_ptr + 0 * H + h_off, mask=mask_h, other=0.0)
        b_z = tl.load(b_hh_ptr + 1 * H + h_off, mask=mask_h, other=0.0)
        b_n = tl.load(b_hh_ptr + 2 * H + h_off, mask=mask_h, other=0.0)
        gh_r += b_r[None, :]
        gh_z += b_z[None, :]
        gh_n += b_n[None, :]

        r = tl.sigmoid(gx_r + gh_r)
        z = tl.sigmoid(gx_z + gh_z)
        n = tl.extra.cuda.libdevice.tanh(gx_n + r * gh_n)
        h = n + z * (h_prev - n)

        base_out = b_off_i64[:, None] * (T * H) + t * H + h_off[None, :]
        tl.store(out_ptr + base_out, h, mask=mask_bh)
        if SAVE_GATES:
            tl.store(save_r_ptr + base_out, r, mask=mask_bh)
            tl.store(save_z_ptr + base_out, z, mask=mask_bh)
            tl.store(save_n_ptr + base_out, n, mask=mask_bh)
            tl.store(save_gh_n_ptr + base_out, gh_n, mask=mask_bh)
        tl.store(h_next_ptr + b_off_i64[:, None] * H + h_off[None, :], h, mask=mask_bh)

    # ------------------------------------------------------------------------
    # GRU backward
    # ------------------------------------------------------------------------

    @triton.autotune(
        configs=_BWD_CONFIGS,
        key=["T", "H", "INPUT_PRECISION"],
        reset_to_zero=["dhidden_ptr"],
        prune_configs_by={
            **_PRUNE_BY_TEMPLATE,
            "early_config_prune": _prune_large_hidden_configs,
        },
    )
    @triton.jit
    def _gru_bwd_kernel(
        hidden_ptr,
        w_r_ptr,
        w_z_ptr,
        w_n_ptr,
        is_init_ptr,
        out_ptr,
        save_r_ptr,
        save_z_ptr,
        save_n_ptr,
        save_gh_n_ptr,
        dout_ptr,
        dh_final_ptr,
        dgates_x_ptr,
        dgates_h_ptr,
        dhidden_ptr,  # [B, T, H] - gradient on per-step hidden values
        B,
        T,
        H: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_K: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b_off = pid * BLOCK_B + tl.arange(0, BLOCK_B)
        b_off_i64 = b_off.to(tl.int64)
        h_off = tl.arange(0, H)
        k_inner = tl.arange(0, BLOCK_K)
        mask_b = b_off < B
        N_K: tl.constexpr = H // BLOCK_K

        dh_next = tl.load(
            dh_final_ptr + b_off_i64[:, None] * H + h_off[None, :],
            mask=mask_b[:, None],
            other=0.0,
        )

        for t_inv in range(T):
            t = T - 1 - t_inv
            base_out = b_off_i64[:, None] * (T * H) + t * H + h_off[None, :]
            dout_t = tl.load(dout_ptr + base_out, mask=mask_b[:, None], other=0.0)
            r = tl.load(save_r_ptr + base_out, mask=mask_b[:, None], other=0.0)
            z = tl.load(save_z_ptr + base_out, mask=mask_b[:, None], other=0.0)
            n = tl.load(save_n_ptr + base_out, mask=mask_b[:, None], other=0.0)
            gh_n = tl.load(save_gh_n_ptr + base_out, mask=mask_b[:, None], other=0.0)

            # See forward kernel: convert uint8 -> i1 once to satisfy newer
            # Triton's ``tl.where`` typing.
            is_init = (
                tl.load(is_init_ptr + b_off_i64 * T + t, mask=mask_b, other=False) != 0
            )
            reset_h = tl.load(
                hidden_ptr + b_off_i64[:, None] * (T * H) + t * H + h_off[None, :],
                mask=mask_b[:, None],
                other=0.0,
            )
            if t == 0:
                h_prev = reset_h
            else:
                h_prev_stored = tl.load(
                    out_ptr + b_off_i64[:, None] * (T * H) + (t - 1) * H + h_off[None, :],
                    mask=mask_b[:, None],
                    other=0.0,
                )
                h_prev = tl.where(is_init[:, None], reset_h, h_prev_stored)

            dh = dout_t + dh_next

            one_minus_z = 1.0 - z
            dn = dh * one_minus_z
            dh_prev_direct = dh * z

            dn_pre = dn * (1.0 - n * n)
            dgh_n = dn_pre * r
            dz_pre = (dh * (h_prev - n)) * z * one_minus_z
            dr_pre = (dn_pre * gh_n) * r * (1.0 - r)

            base_gx = b_off_i64[:, None] * (T * 3 * H) + t * (3 * H)
            tl.store(
                dgates_x_ptr + base_gx + 0 * H + h_off[None, :],
                dr_pre,
                mask=mask_b[:, None],
            )
            tl.store(
                dgates_x_ptr + base_gx + 1 * H + h_off[None, :],
                dz_pre,
                mask=mask_b[:, None],
            )
            tl.store(
                dgates_x_ptr + base_gx + 2 * H + h_off[None, :],
                dn_pre,
                mask=mask_b[:, None],
            )
            tl.store(
                dgates_h_ptr + base_gx + 0 * H + h_off[None, :],
                dr_pre,
                mask=mask_b[:, None],
            )
            tl.store(
                dgates_h_ptr + base_gx + 1 * H + h_off[None, :],
                dz_pre,
                mask=mask_b[:, None],
            )
            tl.store(
                dgates_h_ptr + base_gx + 2 * H + h_off[None, :],
                dgh_n,
                mask=mask_b[:, None],
            )

            dh_prev_total = dh_prev_direct
            for k_iter in tl.static_range(N_K):
                k_off = k_iter * BLOCK_K + k_inner
                base_gx_k = b_off_i64[:, None] * (T * 3 * H) + t * (3 * H) + k_off[None, :]
                w_r_chunk = tl.load(w_r_ptr + k_off[:, None] * H + h_off[None, :])
                dr_chunk = tl.load(
                    dgates_h_ptr + base_gx_k + 0 * H, mask=mask_b[:, None], other=0.0
                )
                dh_prev_total += tl.dot(
                    dr_chunk.to(w_r_chunk.dtype),
                    w_r_chunk,
                    input_precision=INPUT_PRECISION,
                )
                w_z_chunk = tl.load(w_z_ptr + k_off[:, None] * H + h_off[None, :])
                dz_chunk = tl.load(
                    dgates_h_ptr + base_gx_k + 1 * H, mask=mask_b[:, None], other=0.0
                )
                dh_prev_total += tl.dot(
                    dz_chunk.to(w_z_chunk.dtype),
                    w_z_chunk,
                    input_precision=INPUT_PRECISION,
                )
                w_n_chunk = tl.load(w_n_ptr + k_off[:, None] * H + h_off[None, :])
                dn_chunk = tl.load(
                    dgates_h_ptr + base_gx_k + 2 * H, mask=mask_b[:, None], other=0.0
                )
                dh_prev_total += tl.dot(
                    dn_chunk.to(w_n_chunk.dtype),
                    w_n_chunk,
                    input_precision=INPUT_PRECISION,
                )

            # At reset positions, dh_prev flows into dhidden[b, t] not dh_{t-1}.
            tl.atomic_add(
                dhidden_ptr + b_off_i64[:, None] * (T * H) + t * H + h_off[None, :],
                tl.where(is_init[:, None], dh_prev_total, 0.0),
                mask=mask_b[:, None],
            )
            dh_next = tl.where(is_init[:, None], 0.0, dh_prev_total)

        # Fall-off at t=0 goes into dhidden[:, 0, :] (the initial state).
        tl.atomic_add(
            dhidden_ptr + b_off_i64[:, None] * (T * H) + 0 * H + h_off[None, :],
            dh_next,
            mask=mask_b[:, None],
        )

    # ------------------------------------------------------------------------
    # LSTM forward
    # ------------------------------------------------------------------------

    @triton.autotune(
        configs=_FWD_CONFIGS,
        key=["T", "H", "INPUT_PRECISION", "SAVE_GATES"],
        prune_configs_by={
            **_PRUNE_BY_TEMPLATE,
            "early_config_prune": _prune_large_hidden_configs,
        },
    )
    @triton.jit
    def _lstm_fwd_kernel(
        gates_x_ptr,
        hidden_ptr,  # [B, T, H]
        cell_ptr,  # [B, T, H]
        w_i_ptr,
        w_f_ptr,
        w_g_ptr,
        w_o_ptr,
        b_hh_ptr,
        is_init_ptr,
        out_ptr,
        c_out_ptr,
        save_i_ptr,
        save_f_ptr,
        save_g_ptr,
        save_o_ptr,
        save_tanhc_ptr,
        h_final_ptr,
        c_final_ptr,
        B,
        T,
        H: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_K: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        SAVE_GATES: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b_off = pid * BLOCK_B + tl.arange(0, BLOCK_B)
        b_off_i64 = b_off.to(tl.int64)
        h_off = tl.arange(0, H)
        k_inner = tl.arange(0, BLOCK_K)
        mask_b = b_off < B
        N_K: tl.constexpr = H // BLOCK_K

        h = tl.load(
            hidden_ptr + b_off_i64[:, None] * (T * H) + 0 * H + h_off[None, :],
            mask=mask_b[:, None],
            other=0.0,
        )
        c = tl.load(
            cell_ptr + b_off_i64[:, None] * (T * H) + 0 * H + h_off[None, :],
            mask=mask_b[:, None],
            other=0.0,
        )
        b_i = tl.load(b_hh_ptr + 0 * H + h_off)
        b_f = tl.load(b_hh_ptr + 1 * H + h_off)
        b_g = tl.load(b_hh_ptr + 2 * H + h_off)
        b_o = tl.load(b_hh_ptr + 3 * H + h_off)

        for t in range(T):
            base_x = b_off_i64[:, None] * (T * 4 * H) + t * (4 * H)
            gx_i = tl.load(
                gates_x_ptr + base_x + 0 * H + h_off[None, :],
                mask=mask_b[:, None],
                other=0.0,
            )
            gx_f = tl.load(
                gates_x_ptr + base_x + 1 * H + h_off[None, :],
                mask=mask_b[:, None],
                other=0.0,
            )
            gx_g = tl.load(
                gates_x_ptr + base_x + 2 * H + h_off[None, :],
                mask=mask_b[:, None],
                other=0.0,
            )
            gx_o = tl.load(
                gates_x_ptr + base_x + 3 * H + h_off[None, :],
                mask=mask_b[:, None],
                other=0.0,
            )

            # See GRU forward kernel: cast uint8 storage to i1.
            is_init = (
                tl.load(is_init_ptr + b_off_i64 * T + t, mask=mask_b, other=False) != 0
            )
            reset_h = tl.load(
                hidden_ptr + b_off_i64[:, None] * (T * H) + t * H + h_off[None, :],
                mask=mask_b[:, None],
                other=0.0,
            )
            reset_c = tl.load(
                cell_ptr + b_off_i64[:, None] * (T * H) + t * H + h_off[None, :],
                mask=mask_b[:, None],
                other=0.0,
            )
            h = tl.where(is_init[:, None], reset_h, h)
            c = tl.where(is_init[:, None], reset_c, c)

            gh_i = tl.zeros([BLOCK_B, H], dtype=tl.float32)
            gh_f = tl.zeros([BLOCK_B, H], dtype=tl.float32)
            gh_g = tl.zeros([BLOCK_B, H], dtype=tl.float32)
            gh_o = tl.zeros([BLOCK_B, H], dtype=tl.float32)

            for k_iter in tl.static_range(N_K):
                k_off = k_iter * BLOCK_K + k_inner
                if t == 0:
                    h_chunk = tl.load(
                        hidden_ptr + b_off_i64[:, None] * (T * H) + 0 * H + k_off[None, :],
                        mask=mask_b[:, None],
                        other=0.0,
                    )
                else:
                    h_prev_stored = tl.load(
                        out_ptr
                        + b_off_i64[:, None] * (T * H)
                        + (t - 1) * H
                        + k_off[None, :],
                        mask=mask_b[:, None],
                        other=0.0,
                    )
                    reset_chunk = tl.load(
                        hidden_ptr + b_off_i64[:, None] * (T * H) + t * H + k_off[None, :],
                        mask=mask_b[:, None],
                        other=0.0,
                    )
                    h_chunk = tl.where(is_init[:, None], reset_chunk, h_prev_stored)

                w_i_chunk = tl.load(w_i_ptr + k_off[:, None] * H + h_off[None, :])
                h_chunk_w = h_chunk.to(w_i_chunk.dtype)
                gh_i += tl.dot(h_chunk_w, w_i_chunk, input_precision=INPUT_PRECISION)
                w_f_chunk = tl.load(w_f_ptr + k_off[:, None] * H + h_off[None, :])
                gh_f += tl.dot(h_chunk_w, w_f_chunk, input_precision=INPUT_PRECISION)
                w_g_chunk = tl.load(w_g_ptr + k_off[:, None] * H + h_off[None, :])
                gh_g += tl.dot(h_chunk_w, w_g_chunk, input_precision=INPUT_PRECISION)
                w_o_chunk = tl.load(w_o_ptr + k_off[:, None] * H + h_off[None, :])
                gh_o += tl.dot(h_chunk_w, w_o_chunk, input_precision=INPUT_PRECISION)

            gh_i += b_i[None, :]
            gh_f += b_f[None, :]
            gh_g += b_g[None, :]
            gh_o += b_o[None, :]

            # Use explicit intermediate names and split the cell update into
            # two adds. Triton 3.6's frontend can return ``None`` while typing
            # ``c = f * c + i * g`` (4-term inline expression with a self-
            # referential assignment), surfacing as
            # ``AttributeError("'NoneType' object has no attribute 'type'")``
            # when the kernel is compiled from inside a ``torch.compile`` graph.
            i_gate = tl.sigmoid(gx_i + gh_i)
            f_gate = tl.sigmoid(gx_f + gh_f)
            g_gate = tl.extra.cuda.libdevice.tanh(gx_g + gh_g)
            o_gate = tl.sigmoid(gx_o + gh_o)
            c_decayed = f_gate * c
            c_input = i_gate * g_gate
            c = c_decayed + c_input
            tanh_c = tl.extra.cuda.libdevice.tanh(c)
            h = o_gate * tanh_c

            base_out = b_off_i64[:, None] * (T * H) + t * H + h_off[None, :]
            tl.store(out_ptr + base_out, h, mask=mask_b[:, None])
            tl.store(c_out_ptr + base_out, c, mask=mask_b[:, None])
            if SAVE_GATES:
                tl.store(save_i_ptr + base_out, i_gate, mask=mask_b[:, None])
                tl.store(save_f_ptr + base_out, f_gate, mask=mask_b[:, None])
                tl.store(save_g_ptr + base_out, g_gate, mask=mask_b[:, None])
                tl.store(save_o_ptr + base_out, o_gate, mask=mask_b[:, None])
                tl.store(save_tanhc_ptr + base_out, tanh_c, mask=mask_b[:, None])

        tl.store(
            h_final_ptr + b_off_i64[:, None] * H + h_off[None, :], h, mask=mask_b[:, None]
        )
        tl.store(
            c_final_ptr + b_off_i64[:, None] * H + h_off[None, :], c, mask=mask_b[:, None]
        )

    @triton.jit(do_not_specialize=["t"])
    def _lstm_fwd_step_tiled_h_kernel(
        gates_x_ptr,
        hidden_ptr,  # [B, T, H]
        cell_ptr,  # [B, T, H]
        h_prev_ptr,  # [B, H] previous hidden state for this step
        c_prev_ptr,  # [B, H] previous cell state for this step
        w_i_ptr,
        w_f_ptr,
        w_g_ptr,
        w_o_ptr,
        b_hh_ptr,
        is_init_ptr,
        out_ptr,
        c_out_ptr,
        save_i_ptr,
        save_f_ptr,
        save_g_ptr,
        save_o_ptr,
        save_tanhc_ptr,
        h_next_ptr,  # [B, H] next hidden state for this step
        c_next_ptr,  # [B, H] next cell state for this step
        B,
        T,
        H: tl.constexpr,
        t,
        BLOCK_B: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_K: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        SAVE_GATES: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        b_off = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        b_off_i64 = b_off.to(tl.int64)
        h_off = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        k_inner = tl.arange(0, BLOCK_K)
        mask_b = b_off < B
        mask_h = h_off < H
        mask_bh = mask_b[:, None] & mask_h[None, :]
        N_K: tl.constexpr = H // BLOCK_K

        is_init = tl.load(is_init_ptr + b_off_i64 * T + t, mask=mask_b, other=False) != 0
        reset_h = tl.load(
            hidden_ptr + b_off_i64[:, None] * (T * H) + t * H + h_off[None, :],
            mask=mask_bh,
            other=0.0,
        )
        reset_c = tl.load(
            cell_ptr + b_off_i64[:, None] * (T * H) + t * H + h_off[None, :],
            mask=mask_bh,
            other=0.0,
        )
        h_prev = tl.load(
            h_prev_ptr + b_off_i64[:, None] * H + h_off[None, :],
            mask=mask_bh,
            other=0.0,
        )
        c_prev = tl.load(
            c_prev_ptr + b_off_i64[:, None] * H + h_off[None, :],
            mask=mask_bh,
            other=0.0,
        )
        h_prev = tl.where(is_init[:, None], reset_h, h_prev)
        c_prev = tl.where(is_init[:, None], reset_c, c_prev)

        base_x = b_off_i64[:, None] * (T * 4 * H) + t * (4 * H)
        gx_i = tl.load(
            gates_x_ptr + base_x + 0 * H + h_off[None, :],
            mask=mask_bh,
            other=0.0,
        )
        gx_f = tl.load(
            gates_x_ptr + base_x + 1 * H + h_off[None, :],
            mask=mask_bh,
            other=0.0,
        )
        gx_g = tl.load(
            gates_x_ptr + base_x + 2 * H + h_off[None, :],
            mask=mask_bh,
            other=0.0,
        )
        gx_o = tl.load(
            gates_x_ptr + base_x + 3 * H + h_off[None, :],
            mask=mask_bh,
            other=0.0,
        )

        gh_i = tl.zeros([BLOCK_B, BLOCK_H], dtype=tl.float32)
        gh_f = tl.zeros([BLOCK_B, BLOCK_H], dtype=tl.float32)
        gh_g = tl.zeros([BLOCK_B, BLOCK_H], dtype=tl.float32)
        gh_o = tl.zeros([BLOCK_B, BLOCK_H], dtype=tl.float32)

        for k_iter in tl.static_range(N_K):
            k_off = k_iter * BLOCK_K + k_inner
            h_chunk = tl.load(
                h_prev_ptr + b_off_i64[:, None] * H + k_off[None, :],
                mask=mask_b[:, None],
                other=0.0,
            )
            reset_chunk = tl.load(
                hidden_ptr + b_off_i64[:, None] * (T * H) + t * H + k_off[None, :],
                mask=mask_b[:, None],
                other=0.0,
            )
            h_chunk = tl.where(is_init[:, None], reset_chunk, h_chunk)

            w_i_chunk = tl.load(
                w_i_ptr + k_off[:, None] * H + h_off[None, :],
                mask=mask_h[None, :],
                other=0.0,
            )
            h_chunk_w = h_chunk.to(w_i_chunk.dtype)
            gh_i += tl.dot(h_chunk_w, w_i_chunk, input_precision=INPUT_PRECISION)
            w_f_chunk = tl.load(
                w_f_ptr + k_off[:, None] * H + h_off[None, :],
                mask=mask_h[None, :],
                other=0.0,
            )
            gh_f += tl.dot(h_chunk_w, w_f_chunk, input_precision=INPUT_PRECISION)
            w_g_chunk = tl.load(
                w_g_ptr + k_off[:, None] * H + h_off[None, :],
                mask=mask_h[None, :],
                other=0.0,
            )
            gh_g += tl.dot(h_chunk_w, w_g_chunk, input_precision=INPUT_PRECISION)
            w_o_chunk = tl.load(
                w_o_ptr + k_off[:, None] * H + h_off[None, :],
                mask=mask_h[None, :],
                other=0.0,
            )
            gh_o += tl.dot(h_chunk_w, w_o_chunk, input_precision=INPUT_PRECISION)

        b_i = tl.load(b_hh_ptr + 0 * H + h_off, mask=mask_h, other=0.0)
        b_f = tl.load(b_hh_ptr + 1 * H + h_off, mask=mask_h, other=0.0)
        b_g = tl.load(b_hh_ptr + 2 * H + h_off, mask=mask_h, other=0.0)
        b_o = tl.load(b_hh_ptr + 3 * H + h_off, mask=mask_h, other=0.0)
        gh_i += b_i[None, :]
        gh_f += b_f[None, :]
        gh_g += b_g[None, :]
        gh_o += b_o[None, :]

        i_gate = tl.sigmoid(gx_i + gh_i)
        f_gate = tl.sigmoid(gx_f + gh_f)
        g_gate = tl.extra.cuda.libdevice.tanh(gx_g + gh_g)
        o_gate = tl.sigmoid(gx_o + gh_o)
        c_decayed = f_gate * c_prev
        c_input = i_gate * g_gate
        c = c_decayed + c_input
        tanh_c = tl.extra.cuda.libdevice.tanh(c)
        h = o_gate * tanh_c

        base_out = b_off_i64[:, None] * (T * H) + t * H + h_off[None, :]
        tl.store(out_ptr + base_out, h, mask=mask_bh)
        tl.store(c_out_ptr + base_out, c, mask=mask_bh)
        if SAVE_GATES:
            tl.store(save_i_ptr + base_out, i_gate, mask=mask_bh)
            tl.store(save_f_ptr + base_out, f_gate, mask=mask_bh)
            tl.store(save_g_ptr + base_out, g_gate, mask=mask_bh)
            tl.store(save_o_ptr + base_out, o_gate, mask=mask_bh)
            tl.store(save_tanhc_ptr + base_out, tanh_c, mask=mask_bh)
        tl.store(h_next_ptr + b_off_i64[:, None] * H + h_off[None, :], h, mask=mask_bh)
        tl.store(c_next_ptr + b_off_i64[:, None] * H + h_off[None, :], c, mask=mask_bh)

    # ------------------------------------------------------------------------
    # LSTM backward
    # ------------------------------------------------------------------------

    @triton.autotune(
        configs=_BWD_CONFIGS,
        key=["T", "H", "INPUT_PRECISION"],
        reset_to_zero=["dhidden_ptr", "dcell_ptr"],
        prune_configs_by={
            **_PRUNE_BY_TEMPLATE,
            "early_config_prune": _prune_large_hidden_configs,
        },
    )
    @triton.jit
    def _lstm_bwd_kernel(
        hidden_ptr,
        cell_ptr,
        w_i_ptr,
        w_f_ptr,
        w_g_ptr,
        w_o_ptr,
        is_init_ptr,
        out_ptr,
        c_out_ptr,
        save_i_ptr,
        save_f_ptr,
        save_g_ptr,
        save_o_ptr,
        save_tanhc_ptr,
        dout_ptr,
        dc_out_ptr,
        dh_final_ptr,
        dc_final_ptr,
        dgates_x_ptr,
        dgates_h_ptr,
        dhidden_ptr,
        dcell_ptr,
        B,
        T,
        H: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_K: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b_off = pid * BLOCK_B + tl.arange(0, BLOCK_B)
        b_off_i64 = b_off.to(tl.int64)
        h_off = tl.arange(0, H)
        k_inner = tl.arange(0, BLOCK_K)
        mask_b = b_off < B
        N_K: tl.constexpr = H // BLOCK_K

        dh_next = tl.load(
            dh_final_ptr + b_off_i64[:, None] * H + h_off[None, :],
            mask=mask_b[:, None],
            other=0.0,
        )
        dc_next = tl.load(
            dc_final_ptr + b_off_i64[:, None] * H + h_off[None, :],
            mask=mask_b[:, None],
            other=0.0,
        )

        for t_inv in range(T):
            t = T - 1 - t_inv
            base_out = b_off_i64[:, None] * (T * H) + t * H + h_off[None, :]
            dout_t = tl.load(dout_ptr + base_out, mask=mask_b[:, None], other=0.0)
            dc_out_t = tl.load(dc_out_ptr + base_out, mask=mask_b[:, None], other=0.0)
            i = tl.load(save_i_ptr + base_out, mask=mask_b[:, None], other=0.0)
            f = tl.load(save_f_ptr + base_out, mask=mask_b[:, None], other=0.0)
            g = tl.load(save_g_ptr + base_out, mask=mask_b[:, None], other=0.0)
            o = tl.load(save_o_ptr + base_out, mask=mask_b[:, None], other=0.0)
            tanh_c = tl.load(save_tanhc_ptr + base_out, mask=mask_b[:, None], other=0.0)

            # See GRU forward kernel: cast uint8 storage to i1.
            is_init = (
                tl.load(is_init_ptr + b_off_i64 * T + t, mask=mask_b, other=False) != 0
            )
            reset_c = tl.load(
                cell_ptr + b_off_i64[:, None] * (T * H) + t * H + h_off[None, :],
                mask=mask_b[:, None],
                other=0.0,
            )
            if t == 0:
                c_prev = reset_c
            else:
                c_prev_stored = tl.load(
                    c_out_ptr + b_off_i64[:, None] * (T * H) + (t - 1) * H + h_off[None, :],
                    mask=mask_b[:, None],
                    other=0.0,
                )
                c_prev = tl.where(is_init[:, None], reset_c, c_prev_stored)

            dh = dout_t + dh_next
            do = dh * tanh_c
            dc = dc_out_t + dc_next + dh * o * (1.0 - tanh_c * tanh_c)

            df = dc * c_prev
            di = dc * g
            dg = dc * i
            dc_prev_direct = dc * f

            di_pre = di * i * (1.0 - i)
            df_pre = df * f * (1.0 - f)
            dg_pre = dg * (1.0 - g * g)
            do_pre = do * o * (1.0 - o)

            base_gx = b_off_i64[:, None] * (T * 4 * H) + t * (4 * H)
            tl.store(
                dgates_x_ptr + base_gx + 0 * H + h_off[None, :],
                di_pre,
                mask=mask_b[:, None],
            )
            tl.store(
                dgates_x_ptr + base_gx + 1 * H + h_off[None, :],
                df_pre,
                mask=mask_b[:, None],
            )
            tl.store(
                dgates_x_ptr + base_gx + 2 * H + h_off[None, :],
                dg_pre,
                mask=mask_b[:, None],
            )
            tl.store(
                dgates_x_ptr + base_gx + 3 * H + h_off[None, :],
                do_pre,
                mask=mask_b[:, None],
            )
            tl.store(
                dgates_h_ptr + base_gx + 0 * H + h_off[None, :],
                di_pre,
                mask=mask_b[:, None],
            )
            tl.store(
                dgates_h_ptr + base_gx + 1 * H + h_off[None, :],
                df_pre,
                mask=mask_b[:, None],
            )
            tl.store(
                dgates_h_ptr + base_gx + 2 * H + h_off[None, :],
                dg_pre,
                mask=mask_b[:, None],
            )
            tl.store(
                dgates_h_ptr + base_gx + 3 * H + h_off[None, :],
                do_pre,
                mask=mask_b[:, None],
            )

            dh_prev_total = tl.zeros_like(dh_next)
            for k_iter in tl.static_range(N_K):
                k_off = k_iter * BLOCK_K + k_inner
                base_gx_k = b_off_i64[:, None] * (T * 4 * H) + t * (4 * H) + k_off[None, :]
                w_i_chunk = tl.load(w_i_ptr + k_off[:, None] * H + h_off[None, :])
                di_chunk = tl.load(
                    dgates_h_ptr + base_gx_k + 0 * H, mask=mask_b[:, None], other=0.0
                )
                dh_prev_total += tl.dot(
                    di_chunk.to(w_i_chunk.dtype),
                    w_i_chunk,
                    input_precision=INPUT_PRECISION,
                )
                w_f_chunk = tl.load(w_f_ptr + k_off[:, None] * H + h_off[None, :])
                df_chunk = tl.load(
                    dgates_h_ptr + base_gx_k + 1 * H, mask=mask_b[:, None], other=0.0
                )
                dh_prev_total += tl.dot(
                    df_chunk.to(w_f_chunk.dtype),
                    w_f_chunk,
                    input_precision=INPUT_PRECISION,
                )
                w_g_chunk = tl.load(w_g_ptr + k_off[:, None] * H + h_off[None, :])
                dg_chunk = tl.load(
                    dgates_h_ptr + base_gx_k + 2 * H, mask=mask_b[:, None], other=0.0
                )
                dh_prev_total += tl.dot(
                    dg_chunk.to(w_g_chunk.dtype),
                    w_g_chunk,
                    input_precision=INPUT_PRECISION,
                )
                w_o_chunk = tl.load(w_o_ptr + k_off[:, None] * H + h_off[None, :])
                do_chunk = tl.load(
                    dgates_h_ptr + base_gx_k + 3 * H, mask=mask_b[:, None], other=0.0
                )
                dh_prev_total += tl.dot(
                    do_chunk.to(w_o_chunk.dtype),
                    w_o_chunk,
                    input_precision=INPUT_PRECISION,
                )

            tl.atomic_add(
                dhidden_ptr + b_off_i64[:, None] * (T * H) + t * H + h_off[None, :],
                tl.where(is_init[:, None], dh_prev_total, 0.0),
                mask=mask_b[:, None],
            )
            tl.atomic_add(
                dcell_ptr + b_off_i64[:, None] * (T * H) + t * H + h_off[None, :],
                tl.where(is_init[:, None], dc_prev_direct, 0.0),
                mask=mask_b[:, None],
            )
            dh_next = tl.where(is_init[:, None], 0.0, dh_prev_total)
            dc_next = tl.where(is_init[:, None], 0.0, dc_prev_direct)

        tl.atomic_add(
            dhidden_ptr + b_off_i64[:, None] * (T * H) + 0 * H + h_off[None, :],
            dh_next,
            mask=mask_b[:, None],
        )
        tl.atomic_add(
            dcell_ptr + b_off_i64[:, None] * (T * H) + 0 * H + h_off[None, :],
            dc_next,
            mask=mask_b[:, None],
        )


    # ------------------------------------------------------------------------
    # Forward launchers
    # ------------------------------------------------------------------------
    # Fused single-launch for small hidden sizes; per-step hidden-tiled for
    # large H (where a full-hidden program no longer fits in registers /
    # compiles in bounded time). Shared by the forward pass and the recompute
    # backward's forward replay so both follow the same path for a given H.

    def _gru_fwd_launch(
        gates_x,
        hidden_p,
        w_r_t,
        w_z_t,
        w_n_t,
        b_hh_p,
        is_init,
        out,
        save_r,
        save_z,
        save_n,
        save_gh_n,
        B,
        T,
        H_pad,
        input_precision,
        save_gates,
    ):
        """Run the GRU forward recurrence into ``out``; return ``h_final``.

        For ``H_pad >= _FWD_TILED_H_MIN`` the hidden axis is tiled and one
        launch is emitted per time step (the launch boundary is the cross-tile
        barrier that ``h[t + 1]`` needs). Smaller hidden sizes keep the fused
        single-launch path, which holds the recurrent state in registers across
        the whole sequence.
        """
        if H_pad >= _FWD_TILED_H_MIN:

            def grid(meta):
                return (
                    triton.cdiv(B, meta["BLOCK_B"]),
                    triton.cdiv(H_pad, meta["BLOCK_H"]),
                )

            h_prev = hidden_p[:, 0].contiguous()
            h_next = torch.empty_like(h_prev)
            for t in range(T):
                _gru_fwd_step_tiled_h_kernel[grid](
                    gates_x,
                    hidden_p,
                    h_prev,
                    w_r_t,
                    w_z_t,
                    w_n_t,
                    b_hh_p,
                    is_init,
                    out,
                    save_r,
                    save_z,
                    save_n,
                    save_gh_n,
                    h_next,
                    B,
                    T,
                    H=H_pad,
                    t=t,
                    BLOCK_B=_FWD_TILED_BLOCK_B,
                    BLOCK_H=_FWD_TILED_BLOCK_H,
                    BLOCK_K=_FWD_TILED_BLOCK_K,
                    INPUT_PRECISION=input_precision,
                    SAVE_GATES=save_gates,
                    num_warps=_FWD_TILED_NUM_WARPS,
                    num_stages=1,
                )
                h_prev, h_next = h_next, h_prev
            return h_prev

        h_final = torch.empty(B, H_pad, dtype=out.dtype, device=out.device)

        def grid(meta):
            return (triton.cdiv(B, meta["BLOCK_B"]),)

        _gru_fwd_kernel[grid](
            gates_x,
            hidden_p,
            w_r_t,
            w_z_t,
            w_n_t,
            b_hh_p,
            is_init,
            out,
            save_r,
            save_z,
            save_n,
            save_gh_n,
            h_final,
            B,
            T,
            H=H_pad,
            SAVE_GATES=save_gates,
            INPUT_PRECISION=input_precision,
        )
        return h_final

    def _lstm_fwd_launch(
        gates_x,
        hidden_p,
        cell_p,
        w_i_t,
        w_f_t,
        w_g_t,
        w_o_t,
        b_hh_p,
        is_init,
        out,
        c_out,
        save_i,
        save_f,
        save_g,
        save_o,
        save_tanhc,
        B,
        T,
        H_pad,
        input_precision,
        save_gates,
    ):
        """Run the LSTM forward recurrence into ``out`` / ``c_out``.

        Returns ``(h_final, c_final)``. See :func:`_gru_fwd_launch` for the
        fused-vs-tiled rationale.
        """
        if H_pad >= _FWD_TILED_H_MIN:

            def grid(meta):
                return (
                    triton.cdiv(B, meta["BLOCK_B"]),
                    triton.cdiv(H_pad, meta["BLOCK_H"]),
                )

            h_prev = hidden_p[:, 0].contiguous()
            c_prev = cell_p[:, 0].contiguous()
            h_next = torch.empty_like(h_prev)
            c_next = torch.empty_like(c_prev)
            for t in range(T):
                _lstm_fwd_step_tiled_h_kernel[grid](
                    gates_x,
                    hidden_p,
                    cell_p,
                    h_prev,
                    c_prev,
                    w_i_t,
                    w_f_t,
                    w_g_t,
                    w_o_t,
                    b_hh_p,
                    is_init,
                    out,
                    c_out,
                    save_i,
                    save_f,
                    save_g,
                    save_o,
                    save_tanhc,
                    h_next,
                    c_next,
                    B,
                    T,
                    H=H_pad,
                    t=t,
                    BLOCK_B=_FWD_TILED_BLOCK_B,
                    BLOCK_H=_FWD_TILED_BLOCK_H,
                    BLOCK_K=_FWD_TILED_BLOCK_K,
                    INPUT_PRECISION=input_precision,
                    SAVE_GATES=save_gates,
                    num_warps=_FWD_TILED_NUM_WARPS,
                    num_stages=1,
                )
                h_prev, h_next = h_next, h_prev
                c_prev, c_next = c_next, c_prev
            return h_prev, c_prev

        h_final = torch.empty(B, H_pad, dtype=out.dtype, device=out.device)
        c_final = torch.empty_like(h_final)

        def grid(meta):
            return (triton.cdiv(B, meta["BLOCK_B"]),)

        _lstm_fwd_kernel[grid](
            gates_x,
            hidden_p,
            cell_p,
            w_i_t,
            w_f_t,
            w_g_t,
            w_o_t,
            b_hh_p,
            is_init,
            out,
            c_out,
            save_i,
            save_f,
            save_g,
            save_o,
            save_tanhc,
            h_final,
            c_final,
            B,
            T,
            H=H_pad,
            SAVE_GATES=save_gates,
            INPUT_PRECISION=input_precision,
        )
        return h_final, c_final


_MIN_H_PAD_PY = (
    16  # mirror of _MIN_H_PAD; constant so callers outside Triton block can use it
)


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _padded_hidden_size(H: int) -> int:
    """Smallest power-of-two H_pad supported by the kernel autotune grid."""
    return max(_next_pow2(H), _MIN_H_PAD_PY)


def _pad_last(t: torch.Tensor, H: int, H_pad: int) -> torch.Tensor:
    if H == H_pad:
        return t
    pad = list(t.shape)
    pad[-1] = H_pad - H
    return torch.cat([t, t.new_zeros(*pad)], dim=-1)


def _pad_gate_dim(
    t: torch.Tensor, n_gates: int, H: int, H_pad: int, dim: int = 0
) -> torch.Tensor:
    if H == H_pad:
        return t
    shape = list(t.shape)
    assert shape[dim] == n_gates * H
    new_shape = shape[:dim] + [n_gates, H] + shape[dim + 1 :]
    t = t.reshape(new_shape)
    t = _pad_last(t.movedim(dim + 1, -1), H, H_pad).movedim(-1, dim + 1)
    final_shape = shape[:dim] + [n_gates * H_pad] + shape[dim + 1 :]
    return t.reshape(final_shape)


def _pad_w_hh(w_hh: torch.Tensor, n_gates: int, H: int, H_pad: int) -> torch.Tensor:
    if H == H_pad:
        return w_hh
    w_hh = w_hh.view(n_gates, H, H)
    pad_cols = H_pad - H
    w_hh = torch.cat([w_hh, w_hh.new_zeros(n_gates, H, pad_cols)], dim=-1)
    w_hh = torch.cat([w_hh, w_hh.new_zeros(n_gates, pad_cols, H_pad)], dim=-2)
    return w_hh.reshape(n_gates * H_pad, H_pad)


def _unpad_last(t: torch.Tensor, H: int, H_pad: int) -> torch.Tensor:
    if H == H_pad:
        return t
    return t[..., :H].contiguous()


def _unpad_gate_dim(
    t: torch.Tensor, n_gates: int, H: int, H_pad: int, dim: int = 0
) -> torch.Tensor:
    if H == H_pad:
        return t
    shape = list(t.shape)
    new_shape = shape[:dim] + [n_gates, H_pad] + shape[dim + 1 :]
    t = t.reshape(new_shape)
    t = t.narrow(dim + 1, 0, H)
    final_shape = shape[:dim] + [n_gates * H] + shape[dim + 1 :]
    return t.reshape(final_shape).contiguous()


def _unpad_w_hh(t: torch.Tensor, n_gates: int, H: int, H_pad: int) -> torch.Tensor:
    if H == H_pad:
        return t
    t = t.view(n_gates, H_pad, H_pad)
    t = t[:, :H, :H].contiguous()
    return t.reshape(n_gates * H, H)


def _gru_split_w_hh(
    w_hh: torch.Tensor, H: int, H_pad: int, compute_dtype: torch.dtype
) -> tuple[torch.Tensor, ...]:
    """Return ``(w_r_t, w_z_t, w_n_t, w_r, w_z, w_n)`` from a padded ``w_hh``.

    Factored out so the recompute backward can rebuild the per-gate weight
    views from the saved un-padded ``w_hh`` without duplicating the math.
    """
    w_hh_p = _pad_w_hh(w_hh, 3, H, H_pad)
    w_hh_c = w_hh_p.to(compute_dtype)
    w_hh_c3 = w_hh_c.view(3, H_pad, H_pad)
    return (
        w_hh_c3[0].t().contiguous(),
        w_hh_c3[1].t().contiguous(),
        w_hh_c3[2].t().contiguous(),
        w_hh_c3[0].contiguous(),
        w_hh_c3[1].contiguous(),
        w_hh_c3[2].contiguous(),
    )


class _GRUFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        hidden,
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        is_init,
        compute_dtype,
        save_gates,
        input_precision,
    ):
        if not _has_triton:
            raise RuntimeError(
                "Triton is not available. Install triton or use recurrent_backend='pad'/'scan'."
            )
        B, T, I_in = x.shape
        H = hidden.shape[-1]
        H_pad = _padded_hidden_size(H)

        hidden_p = _pad_last(hidden, H, H_pad).contiguous()
        b_ih_p = _pad_gate_dim(b_ih, 3, H, H_pad, dim=0)
        b_hh_p = _pad_gate_dim(b_hh, 3, H, H_pad, dim=0)
        w_ih_p = _pad_gate_dim(w_ih, 3, H, H_pad, dim=0)

        # Flat ``if`` rather than a context manager: compiled_autograd sees
        # a context manager generator as a side-effecting region and either
        # graph-breaks or recompiles per (prev, want) flag combination.
        _prev_tf32 = _maybe_enable_tf32(input_precision)
        gates_x = (
            F.linear(x.reshape(-1, I_in), w_ih_p, b_ih_p)
            .view(B, T, 3 * H_pad)
            .contiguous()
        )
        _restore_tf32(_prev_tf32)

        w_r_t, w_z_t, w_n_t, w_r, w_z, w_n = _gru_split_w_hh(
            w_hh, H, H_pad, compute_dtype
        )

        out = torch.empty(B, T, H_pad, dtype=x.dtype, device=x.device)
        # See ``_LSTMFn.forward`` for the rationale: under ``save_gates=False``
        # the kernel's gate-save stores are dead-code-eliminated via the
        # ``SAVE_GATES`` constexpr, so a single 4-byte placeholder satisfies
        # the pointer args without polluting the CUDA caching allocator /
        # cudagraph private pool.
        if save_gates:
            save_r = torch.empty_like(out)
            save_z = torch.empty_like(out)
            save_n = torch.empty_like(out)
            save_gh_n = torch.empty_like(out)
        else:
            _placeholder = torch.empty(1, dtype=x.dtype, device=x.device)
            save_r = save_z = save_n = save_gh_n = _placeholder

        h_final = _gru_fwd_launch(
            gates_x,
            hidden_p,
            w_r_t,
            w_z_t,
            w_n_t,
            b_hh_p,
            is_init,
            out,
            save_r,
            save_z,
            save_n,
            save_gh_n,
            B,
            T,
            H_pad,
            input_precision,
            save_gates,
        )

        if not save_gates:
            # Drop the per-step gate buffers (save_r/z/n/gh_n) and the per-gate
            # weight slices from the saved set. Backward replays the forward
            # kernel into fresh scratch buffers to recover them.
            ctx.save_for_backward(
                x,
                hidden_p,
                w_ih,
                w_hh,
                b_ih_p,
                b_hh_p,
                is_init,
                out,
                w_ih_p,
            )
        else:
            ctx.save_for_backward(
                x,
                hidden_p,
                w_ih,
                w_hh,
                is_init,
                out,
                save_r,
                save_z,
                save_n,
                save_gh_n,
                w_r,
                w_z,
                w_n,
                w_ih_p,
            )
        ctx.shapes = (B, T, I_in, H, H_pad)
        ctx.recompute = not save_gates
        ctx.compute_dtype = compute_dtype
        ctx.input_precision = input_precision
        return _unpad_last(out, H, H_pad), _unpad_last(h_final, H, H_pad)

    @staticmethod
    def backward(ctx, dout, dh_final):
        B, T, I_in, H, H_pad = ctx.shapes
        input_precision = ctx.input_precision
        if ctx.recompute:
            (
                x,
                hidden_p,
                w_ih,
                w_hh,
                b_ih_p,
                b_hh_p,
                is_init,
                out,
                w_ih_p,
            ) = ctx.saved_tensors
            w_r_t, w_z_t, w_n_t, w_r, w_z, w_n = _gru_split_w_hh(
                w_hh, H, H_pad, ctx.compute_dtype
            )
            _prev_tf32 = _maybe_enable_tf32(input_precision)
            gates_x = (
                F.linear(x.reshape(-1, I_in), w_ih_p, b_ih_p)
                .view(B, T, 3 * H_pad)
                .contiguous()
            )
            _restore_tf32(_prev_tf32)
            save_r = torch.empty_like(out)
            save_z = torch.empty_like(out)
            save_n = torch.empty_like(out)
            save_gh_n = torch.empty_like(out)
            # Scratch ``out`` so the saved ``out`` tensor's version counter
            # stays untouched. Routing through ``_gru_fwd_launch`` replays large
            # hidden sizes via the per-step tiled kernel, like the forward pass.
            out_scratch = torch.empty_like(out)
            _gru_fwd_launch(
                gates_x,
                hidden_p,
                w_r_t,
                w_z_t,
                w_n_t,
                b_hh_p,
                is_init,
                out_scratch,
                save_r,
                save_z,
                save_n,
                save_gh_n,
                B,
                T,
                H_pad,
                input_precision,
                True,
            )
        else:
            (
                x,
                hidden_p,
                w_ih,
                w_hh,
                is_init,
                out,
                save_r,
                save_z,
                save_n,
                save_gh_n,
                w_r,
                w_z,
                w_n,
                w_ih_p,
            ) = ctx.saved_tensors

        dout_p = _pad_last(dout.contiguous(), H, H_pad).contiguous()
        dh_final_p = _pad_last(dh_final.contiguous(), H, H_pad).contiguous()

        dgates_x = torch.empty(B, T, 3 * H_pad, dtype=x.dtype, device=x.device)
        dgates_h = torch.empty_like(dgates_x)
        dhidden_p = torch.zeros_like(hidden_p)

        def grid(meta):
            return (triton.cdiv(B, meta["BLOCK_B"]),)

        _gru_bwd_kernel[grid](
            hidden_p,
            w_r,
            w_z,
            w_n,
            is_init,
            out,
            save_r,
            save_z,
            save_n,
            save_gh_n,
            dout_p,
            dh_final_p,
            dgates_x,
            dgates_h,
            dhidden_p,
            B,
            T,
            H=H_pad,
            INPUT_PRECISION=input_precision,
        )

        # h_prev[b, t] for the dW_hh computation.
        h_prev_all = torch.empty_like(out)
        h_prev_all[:, 0] = hidden_p[:, 0]
        if T > 1:
            h_prev_all[:, 1:] = out[:, :-1]
        h_prev_all = torch.where(is_init.unsqueeze(-1), hidden_p, h_prev_all)

        dgates_h_flat = dgates_h.reshape(B * T, 3 * H_pad)
        h_prev_flat = h_prev_all.reshape(B * T, H_pad)
        dgates_x_flat = dgates_x.reshape(B * T, 3 * H_pad)
        x_flat = x.reshape(B * T, I_in)
        # See forward for why this is a flat ``if`` rather than a context
        # manager. compiled_autograd traces this backward.
        _prev_tf32 = _maybe_enable_tf32(input_precision)
        dW_hh_p = dgates_h_flat.t() @ h_prev_flat
        dW_ih_p = dgates_x_flat.t() @ x_flat
        dx = (dgates_x_flat @ w_ih_p).view(B, T, I_in)
        _restore_tf32(_prev_tf32)
        db_hh_p = dgates_h_flat.sum(0)
        db_ih_p = dgates_x_flat.sum(0)

        dhidden = _unpad_last(dhidden_p, H, H_pad)
        dW_hh = _unpad_w_hh(dW_hh_p, 3, H, H_pad)
        db_hh = _unpad_gate_dim(db_hh_p, 3, H, H_pad, dim=0)
        dW_ih = _unpad_gate_dim(dW_ih_p, 3, H, H_pad, dim=0)
        db_ih = _unpad_gate_dim(db_ih_p, 3, H, H_pad, dim=0)

        return dx, dhidden, dW_ih, dW_hh, db_ih, db_hh, None, None, None, None


def _lstm_split_w_hh(
    w_hh: torch.Tensor, H: int, H_pad: int, compute_dtype: torch.dtype
) -> tuple[torch.Tensor, ...]:
    """Return ``(w_i_t, w_f_t, w_g_t, w_o_t, w_i, w_f, w_g, w_o)`` from ``w_hh``.

    Same role as :func:`_gru_split_w_hh` for LSTM's four gates.
    """
    w_hh_p = _pad_w_hh(w_hh, 4, H, H_pad)
    w_hh_c = w_hh_p.to(compute_dtype)
    w_hh_c4 = w_hh_c.view(4, H_pad, H_pad)
    return (
        w_hh_c4[0].t().contiguous(),
        w_hh_c4[1].t().contiguous(),
        w_hh_c4[2].t().contiguous(),
        w_hh_c4[3].t().contiguous(),
        w_hh_c4[0].contiguous(),
        w_hh_c4[1].contiguous(),
        w_hh_c4[2].contiguous(),
        w_hh_c4[3].contiguous(),
    )


class _LSTMFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        hidden,
        cell,
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        is_init,
        compute_dtype,
        save_gates,
        input_precision,
    ):
        if not _has_triton:
            raise RuntimeError(
                "Triton is not available. Install triton or use recurrent_backend='pad'/'scan'."
            )
        B, T, I_in = x.shape
        H = hidden.shape[-1]
        H_pad = _padded_hidden_size(H)

        hidden_p = _pad_last(hidden, H, H_pad).contiguous()
        cell_p = _pad_last(cell, H, H_pad).contiguous()
        b_ih_p = _pad_gate_dim(b_ih, 4, H, H_pad, dim=0)
        b_hh_p = _pad_gate_dim(b_hh, 4, H, H_pad, dim=0)
        w_ih_p = _pad_gate_dim(w_ih, 4, H, H_pad, dim=0)

        # See GRU forward for the flat-``if`` rationale.
        _prev_tf32 = _maybe_enable_tf32(input_precision)
        gates_x = (
            F.linear(x.reshape(-1, I_in), w_ih_p, b_ih_p)
            .view(B, T, 4 * H_pad)
            .contiguous()
        )
        _restore_tf32(_prev_tf32)

        w_i_t, w_f_t, w_g_t, w_o_t, w_i, w_f, w_g, w_o = _lstm_split_w_hh(
            w_hh, H, H_pad, compute_dtype
        )

        out = torch.empty(B, T, H_pad, dtype=x.dtype, device=x.device)
        c_out = torch.empty_like(out)
        # Gate-save buffers are only ever read by the backward kernel. When
        # ``save_gates`` is False (recompute mode, no-grad call, or no input
        # tracking grad) the forward kernel's ``tl.store(save_*)`` lines are
        # dead-code-eliminated via ``SAVE_GATES: tl.constexpr`` -- so a single
        # 4-byte placeholder is enough to satisfy the kernel's pointer
        # arguments. Avoiding the 5 ``[B, T, H_pad]`` allocations keeps them
        # out of the CUDA caching allocator / cudagraph private pool.
        if save_gates:
            save_i = torch.empty_like(out)
            save_f = torch.empty_like(out)
            save_g = torch.empty_like(out)
            save_o = torch.empty_like(out)
            save_tanhc = torch.empty_like(out)
        else:
            _placeholder = torch.empty(1, dtype=x.dtype, device=x.device)
            save_i = save_f = save_g = save_o = save_tanhc = _placeholder
        h_final, c_final = _lstm_fwd_launch(
            gates_x,
            hidden_p,
            cell_p,
            w_i_t,
            w_f_t,
            w_g_t,
            w_o_t,
            b_hh_p,
            is_init,
            out,
            c_out,
            save_i,
            save_f,
            save_g,
            save_o,
            save_tanhc,
            B,
            T,
            H_pad,
            input_precision,
            save_gates,
        )

        if not save_gates:
            ctx.save_for_backward(
                x,
                hidden_p,
                cell_p,
                w_ih,
                w_hh,
                b_ih_p,
                b_hh_p,
                is_init,
                out,
                c_out,
                w_ih_p,
            )
        else:
            ctx.save_for_backward(
                x,
                hidden_p,
                cell_p,
                w_ih,
                w_hh,
                is_init,
                out,
                c_out,
                save_i,
                save_f,
                save_g,
                save_o,
                save_tanhc,
                w_i,
                w_f,
                w_g,
                w_o,
                w_ih_p,
            )
        ctx.shapes = (B, T, I_in, H, H_pad)
        ctx.recompute = not save_gates
        ctx.compute_dtype = compute_dtype
        ctx.input_precision = input_precision
        return (
            _unpad_last(out, H, H_pad),
            _unpad_last(c_out, H, H_pad),
            _unpad_last(h_final, H, H_pad),
            _unpad_last(c_final, H, H_pad),
        )

    @staticmethod
    def backward(ctx, dout, dc_out, dh_final, dc_final):
        B, T, I_in, H, H_pad = ctx.shapes
        input_precision = ctx.input_precision
        if ctx.recompute:
            (
                x,
                hidden_p,
                cell_p,
                w_ih,
                w_hh,
                b_ih_p,
                b_hh_p,
                is_init,
                out,
                c_out,
                w_ih_p,
            ) = ctx.saved_tensors
            (
                w_i_t,
                w_f_t,
                w_g_t,
                w_o_t,
                w_i,
                w_f,
                w_g,
                w_o,
            ) = _lstm_split_w_hh(w_hh, H, H_pad, ctx.compute_dtype)
            _prev_tf32 = _maybe_enable_tf32(input_precision)
            gates_x = (
                F.linear(x.reshape(-1, I_in), w_ih_p, b_ih_p)
                .view(B, T, 4 * H_pad)
                .contiguous()
            )
            _restore_tf32(_prev_tf32)
            save_i = torch.empty_like(out)
            save_f = torch.empty_like(out)
            save_g = torch.empty_like(out)
            save_o = torch.empty_like(out)
            save_tanhc = torch.empty_like(out)
            # Scratch ``out`` / ``c_out`` keep the saved tensors' version
            # counters untouched; routing through ``_lstm_fwd_launch`` replays
            # large hidden sizes via the per-step tiled kernel, like forward.
            out_scratch = torch.empty_like(out)
            c_out_scratch = torch.empty_like(out)
            _lstm_fwd_launch(
                gates_x,
                hidden_p,
                cell_p,
                w_i_t,
                w_f_t,
                w_g_t,
                w_o_t,
                b_hh_p,
                is_init,
                out_scratch,
                c_out_scratch,
                save_i,
                save_f,
                save_g,
                save_o,
                save_tanhc,
                B,
                T,
                H_pad,
                input_precision,
                True,
            )
        else:
            (
                x,
                hidden_p,
                cell_p,
                w_ih,
                w_hh,
                is_init,
                out,
                c_out,
                save_i,
                save_f,
                save_g,
                save_o,
                save_tanhc,
                w_i,
                w_f,
                w_g,
                w_o,
                w_ih_p,
            ) = ctx.saved_tensors

        dout_p = _pad_last(dout.contiguous(), H, H_pad).contiguous()
        dc_out_p = _pad_last(dc_out.contiguous(), H, H_pad).contiguous()
        dh_final_p = _pad_last(dh_final.contiguous(), H, H_pad).contiguous()
        dc_final_p = _pad_last(dc_final.contiguous(), H, H_pad).contiguous()
        dgates_x = torch.empty(B, T, 4 * H_pad, dtype=x.dtype, device=x.device)
        dgates_h = torch.empty_like(dgates_x)
        dhidden_p = torch.zeros_like(hidden_p)
        dcell_p = torch.zeros_like(cell_p)

        def grid(meta):
            return (triton.cdiv(B, meta["BLOCK_B"]),)

        _lstm_bwd_kernel[grid](
            hidden_p,
            cell_p,
            w_i,
            w_f,
            w_g,
            w_o,
            is_init,
            out,
            c_out,
            save_i,
            save_f,
            save_g,
            save_o,
            save_tanhc,
            dout_p,
            dc_out_p,
            dh_final_p,
            dc_final_p,
            dgates_x,
            dgates_h,
            dhidden_p,
            dcell_p,
            B,
            T,
            H=H_pad,
            INPUT_PRECISION=input_precision,
        )

        h_prev_all = torch.empty_like(out)
        h_prev_all[:, 0] = hidden_p[:, 0]
        if T > 1:
            h_prev_all[:, 1:] = out[:, :-1]
        h_prev_all = torch.where(is_init.unsqueeze(-1), hidden_p, h_prev_all)

        dgates_h_flat = dgates_h.reshape(B * T, 4 * H_pad)
        h_prev_flat = h_prev_all.reshape(B * T, H_pad)
        dgates_x_flat = dgates_x.reshape(B * T, 4 * H_pad)
        x_flat = x.reshape(B * T, I_in)
        # See GRU forward for the flat-``if`` rationale.
        _prev_tf32 = _maybe_enable_tf32(input_precision)
        dW_hh_p = dgates_h_flat.t() @ h_prev_flat
        dW_ih_p = dgates_x_flat.t() @ x_flat
        dx = (dgates_x_flat @ w_ih_p).view(B, T, I_in)
        _restore_tf32(_prev_tf32)
        db_hh_p = dgates_h_flat.sum(0)
        db_ih_p = dgates_x_flat.sum(0)

        dhidden = _unpad_last(dhidden_p, H, H_pad)
        dcell = _unpad_last(dcell_p, H, H_pad)
        dW_hh = _unpad_w_hh(dW_hh_p, 4, H, H_pad)
        db_hh = _unpad_gate_dim(db_hh_p, 4, H, H_pad, dim=0)
        dW_ih = _unpad_gate_dim(dW_ih_p, 4, H, H_pad, dim=0)
        db_ih = _unpad_gate_dim(db_ih_p, 4, H, H_pad, dim=0)

        return (
            dx,
            dhidden,
            dcell,
            dW_ih,
            dW_hh,
            db_ih,
            db_hh,
            None,
            None,
            None,
            None,
        )


def _resolve_save_gates(recompute: bool, *tensors: torch.Tensor) -> bool:
    """Decide whether the Triton forward kernel must store per-step gate buffers.

    Returning ``False`` lets the kernel's ``SAVE_GATES`` constexpr drop the
    ``tl.store(save_*)`` lines and lets the python wrapper skip the
    corresponding ``torch.empty_like(out)`` allocations entirely.

    Must be called *outside* ``torch.autograd.Function.apply`` because
    ``torch.is_grad_enabled()`` is always ``False`` inside ``forward`` even
    during differentiable calls.
    """
    if recompute:
        return False
    if not torch.is_grad_enabled():
        return False
    return any(t is not None and t.requires_grad for t in tensors)


def gru_triton(
    x: torch.Tensor,
    hidden: torch.Tensor,
    w_ih: torch.Tensor,
    w_hh: torch.Tensor,
    b_ih: torch.Tensor,
    b_hh: torch.Tensor,
    is_init: torch.Tensor,
    compute_dtype: torch.dtype = torch.float32,
    *,
    recompute: bool = False,
    input_precision: RecurrentMatmulPrecisionUserMode | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-layer GRU forward with reset, autograd-aware.

    Args:
        x: ``[B, T, input_size]`` fp32 inputs.
        hidden: ``[B, T, hidden_size]`` per-step reset hidden values. ``hidden[:, 0]``
            is also the initial state.
        w_ih: ``[3*H, I]`` fp32 weights.
        w_hh: ``[3*H, H]`` fp32 weights.
        b_ih: ``[3*H]`` fp32 biases.
        b_hh: ``[3*H]`` fp32 biases.
        is_init: ``[B, T]`` bool reset mask.
        compute_dtype: matmul operand dtype. fp32 keeps weights in fp32 (the
            ``input_precision`` argument then dictates whether ``tl.dot`` runs
            IEEE FP32, TF32 or TF32x3). bf16 casts weights to bf16 (wider SMEM
            margin, ~7-bit mantissa, ``input_precision`` ignored).
        recompute: when ``True``, drop the per-step gate buffers
            ``save_r/z/n/gh_n`` from the saved-for-backward set and recompute
            them by replaying the forward kernel during backward. Trades a
            small extra fwd-pass cost for a substantial reduction in backward
            activation memory. The same drop applies automatically when no
            input requires grad or the call is under :func:`torch.no_grad` --
            backward cannot run in those cases, so the forward never allocates
            the gate-save buffers in the first place.
        input_precision: precision passed to ``tl.dot``. Concrete modes
            ``"ieee"``, ``"tf32"``, ``"tf32x3"``; GPU-aware presets
            ``"fast"`` (Ampere+ -> ``"tf32"``, else ``"ieee"``) and
            ``"high-prec"`` (Ampere+ -> ``"tf32x3"``, else ``"ieee"``); or
            ``None`` / ``"auto"`` to derive from
            :func:`torchrl.modules.get_recurrent_matmul_precision`.

    Returns:
        ``(out, h_final)`` where ``out`` is ``[B, T, H]`` and ``h_final`` is ``[B, H]``.
    """
    resolved = _resolve_precision(input_precision)
    save_gates = _resolve_save_gates(recompute, x, hidden, w_ih, w_hh, b_ih, b_hh)
    return _GRUFn.apply(
        x,
        hidden,
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        is_init,
        compute_dtype,
        save_gates,
        resolved,
    )


def lstm_triton(
    x: torch.Tensor,
    hidden: torch.Tensor,
    cell: torch.Tensor,
    w_ih: torch.Tensor,
    w_hh: torch.Tensor,
    b_ih: torch.Tensor,
    b_hh: torch.Tensor,
    is_init: torch.Tensor,
    compute_dtype: torch.dtype = torch.float32,
    *,
    recompute: bool = False,
    input_precision: RecurrentMatmulPrecisionUserMode | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-layer LSTM forward with reset, autograd-aware.

    See :func:`gru_triton` for argument conventions. ``cell`` carries the
    per-step reset cell values (``c0`` semantics). ``recompute`` has the same
    semantics as in :func:`gru_triton`: drops the five LSTM gate buffers
    (``save_i/f/g/o/save_tanhc``) from saved state in exchange for an extra
    forward replay in backward. The drop also happens automatically under
    :func:`torch.no_grad` / when no input tracks grad. ``input_precision`` has
    the same semantics as in :func:`gru_triton`.

    Returns:
        ``(h_steps, c_steps, h_final, c_final)``.
    """
    resolved = _resolve_precision(input_precision)
    save_gates = _resolve_save_gates(recompute, x, hidden, cell, w_ih, w_hh, b_ih, b_hh)
    return _LSTMFn.apply(
        x,
        hidden,
        cell,
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        is_init,
        compute_dtype,
        save_gates,
        resolved,
    )
