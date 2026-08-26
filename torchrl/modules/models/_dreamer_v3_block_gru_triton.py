# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Fused Triton recurrence for :class:`DreamerV3BlockGRU`.

The kernels in this file are deliberately model-specific.  They keep the
hidden carry in one Triton program across the complete time dimension, while
the Python autograd wrapper performs the large parameter reductions as flat
PyTorch contractions.
"""
from __future__ import annotations

import importlib.metadata

import torch
from packaging import version
from torch import nn
from torch.nn import functional as F


def _check_triton_available() -> bool:
    try:
        triton_version = importlib.metadata.version("triton")
    except importlib.metadata.PackageNotFoundError:
        return False
    # Batched tl.dot is used for the independent block contractions.
    return version.parse(triton_version) >= version.parse("3.3")


_has_triton = _check_triton_available()


def _activation_code(activation: nn.Module) -> int:
    if isinstance(activation, nn.SiLU):
        return 0
    if isinstance(activation, nn.Tanh):
        return 1
    if isinstance(activation, nn.ReLU):
        return 2
    raise ValueError(
        "The DreamerV3 block-GRU Triton backend supports nn.SiLU, nn.Tanh, "
        f"and nn.ReLU, got {type(activation).__name__}."
    )


def _activation(value: torch.Tensor, code: int) -> torch.Tensor:
    if code == 0:
        return F.silu(value)
    if code == 1:
        return value.tanh()
    return value.relu()


def _block_weight_grad(
    value: torch.Tensor, grad_output: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    num_blocks, block_in, block_out = weight.shape
    value = value.reshape(-1, num_blocks, block_in).float()
    grad_output = grad_output.reshape(-1, num_blocks, block_out).float()
    return torch.einsum("nbi,nbo->bio", value, grad_output)


if _has_triton:
    import triton
    import triton.language as tl

    _CONFIGS = [
        triton.Config({"BLOCK_B": 1, "BLOCK_K": 16}, num_warps=1),
        triton.Config({"BLOCK_B": 2, "BLOCK_K": 16}, num_warps=2),
        triton.Config({"BLOCK_B": 4, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_B": 8, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_B": 8, "BLOCK_K": 64}, num_warps=8),
    ]

    def _prune_configs(configs, named_args, **kwargs):
        h_pad = kwargs.get("H_PAD") or named_args["H_PAD"]
        p_pad = kwargs.get("P_PAD") or named_args["P_PAD"]
        d_pad = kwargs.get("D_PAD") or named_args["D_PAD"]
        limit = min(h_pad, p_pad, d_pad)
        return [
            config
            for config in configs
            if config.kwargs["BLOCK_K"] <= limit
            and h_pad % config.kwargs["BLOCK_K"] == 0
            and p_pad % config.kwargs["BLOCK_K"] == 0
            and d_pad % config.kwargs["BLOCK_K"] == 0
        ]

    @triton.jit
    def _activate(value, ACTIVATION: tl.constexpr):
        if ACTIVATION == 0:
            return value * tl.sigmoid(value)
        if ACTIVATION == 1:
            return tl.extra.cuda.libdevice.tanh(value)
        return tl.maximum(value, 0.0)

    @triton.jit
    def _activation_grad(value, ACTIVATION: tl.constexpr):
        if ACTIVATION == 0:
            sigmoid = tl.sigmoid(value)
            return sigmoid * (1.0 + value * (1.0 - sigmoid))
        if ACTIVATION == 1:
            output = tl.extra.cuda.libdevice.tanh(value)
            return 1.0 - output * output
        return (value > 0.0).to(tl.float32)

    @triton.autotune(
        configs=_CONFIGS,
        key=[
            "B",
            "T",
            "H",
            "P",
            "NUM_BLOCKS",
            "NUM_LAYERS",
            "ACTIVATION",
            "COMPUTE_BF16",
            "SAVE_STATE",
        ],
        prune_configs_by={
            "perf_model": None,
            "top_k": None,
            "early_config_prune": _prune_configs,
        },
    )
    @triton.jit
    def _block_gru_fwd_kernel(
        projected_contribution_ptr,
        initial_ptr,
        is_init_ptr,
        hidden_weight_ptr,
        hidden_bias_ptr,
        hidden_norm_ptr,
        first_recurrent_weight_ptr,
        later_weight_ptr,
        dynamic_bias_ptr,
        dynamic_norm_ptr,
        gate_weight_ptr,
        gate_bias_ptr,
        output_ptr,
        final_ptr,
        hidden_normalized_ptr,
        hidden_inv_rms_ptr,
        dynamic_normalized_ptr,
        dynamic_inv_rms_ptr,
        gate_state_ptr,
        B,
        T,
        H: tl.constexpr,
        P: tl.constexpr,
        H_PAD: tl.constexpr,
        P_PAD: tl.constexpr,
        D: tl.constexpr,
        D_PAD: tl.constexpr,
        G_PAD: tl.constexpr,
        K_REC_PAD: tl.constexpr,
        NUM_BLOCKS: tl.constexpr,
        NUM_LAYERS: tl.constexpr,
        NORM_EPS: tl.constexpr,
        UPDATE_BIAS: tl.constexpr,
        ACTIVATION: tl.constexpr,
        COMPUTE_BF16: tl.constexpr,
        SAVE_STATE: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid * BLOCK_B + tl.arange(0, BLOCK_B)
        b64 = b.to(tl.int64)
        mask_b = b < B
        h_off = tl.arange(0, H_PAD)
        p_off = tl.arange(0, P_PAD)
        d_off = tl.arange(0, D_PAD)
        k_off = tl.arange(0, BLOCK_K)
        for t in range(T):
            reset = tl.load(is_init_ptr + b64 * T + t, mask=mask_b, other=False) != 0
            if t == 0:
                previous = tl.load(
                    initial_ptr + b64[:, None] * H + h_off[None, :],
                    mask=mask_b[:, None] & (h_off[None, :] < H),
                    other=0.0,
                )
            else:
                previous = tl.load(
                    output_ptr + (b64[:, None] * T + t - 1) * H + h_off[None, :],
                    mask=mask_b[:, None] & (h_off[None, :] < H),
                    other=0.0,
                )
            previous = tl.where(reset[:, None], 0.0, previous)

            hidden_pre = tl.zeros([BLOCK_B, P_PAD], tl.float32)
            for k_iter in tl.static_range(tl.cdiv(H_PAD, BLOCK_K)):
                hk = k_iter * BLOCK_K + k_off
                h_chunk = tl.where(hk[None, :] < H, previous[:, hk], 0.0)
                weight = tl.load(
                    hidden_weight_ptr + hk[:, None] * P_PAD + p_off[None, :],
                    mask=(hk[:, None] < H_PAD) & (p_off[None, :] < P_PAD),
                    other=0.0,
                )
                if COMPUTE_BF16:
                    h_chunk = h_chunk.to(tl.bfloat16)
                hidden_pre += tl.dot(h_chunk, weight, input_precision="ieee")
            hidden_bias = tl.load(hidden_bias_ptr + p_off, mask=p_off < P, other=0.0)
            hidden_pre += hidden_bias[None, :]
            hidden_sumsq = tl.sum(hidden_pre * hidden_pre, axis=1) / P
            hidden_inv = tl.rsqrt(hidden_sumsq + NORM_EPS)
            hidden_scale = tl.load(hidden_norm_ptr + p_off, mask=p_off < P, other=0.0)
            hidden_xhat = hidden_pre * hidden_inv[:, None]
            hidden_value = _activate(hidden_xhat * hidden_scale[None, :], ACTIVATION)
            hidden_value = tl.where(p_off[None, :] < P, hidden_value, 0.0)

            if SAVE_STATE:
                hidden_base = (b64[:, None] * T + t) * P + p_off[None, :]
                tl.store(
                    hidden_normalized_ptr + hidden_base,
                    hidden_xhat,
                    mask=mask_b[:, None] & (p_off[None, :] < P),
                )
                tl.store(hidden_inv_rms_ptr + b64 * T + t, hidden_inv, mask=mask_b)

            for layer in tl.static_range(NUM_LAYERS):
                sumsq = tl.zeros([BLOCK_B], tl.float32)
                for block in tl.static_range(NUM_BLOCKS):
                    pre = tl.zeros([BLOCK_B, D_PAD], tl.float32)
                    if layer == 0:
                        for k_iter in tl.static_range(tl.cdiv(D_PAD, BLOCK_K)):
                            kk = k_iter * BLOCK_K + k_off
                            carry_index = block * D + kk
                            recurrent = tl.where(
                                kk[None, :] < D, previous[:, carry_index], 0.0
                            )
                            weight = tl.load(
                                first_recurrent_weight_ptr
                                + block * K_REC_PAD * D_PAD
                                + kk[:, None] * D_PAD
                                + d_off[None, :],
                                mask=(kk[:, None] < K_REC_PAD)
                                & (d_off[None, :] < D_PAD),
                                other=0.0,
                            )
                            if COMPUTE_BF16:
                                recurrent = recurrent.to(tl.bfloat16)
                            pre += tl.dot(recurrent, weight, input_precision="ieee")
                        for k_iter in tl.static_range(tl.cdiv(P_PAD, BLOCK_K)):
                            kk = k_iter * BLOCK_K + k_off
                            recurrent = tl.where(
                                kk[None, :] < P, hidden_value[:, kk], 0.0
                            )
                            weight = tl.load(
                                first_recurrent_weight_ptr
                                + block * K_REC_PAD * D_PAD
                                + (D + kk)[:, None] * D_PAD
                                + d_off[None, :],
                                mask=((D + kk)[:, None] < K_REC_PAD)
                                & (d_off[None, :] < D_PAD),
                                other=0.0,
                            )
                            if COMPUTE_BF16:
                                recurrent = recurrent.to(tl.bfloat16)
                            pre += tl.dot(recurrent, weight, input_precision="ieee")
                        projected_base = (
                            (b64[:, None] * T + t) * H + block * D + d_off[None, :]
                        )
                        pre += tl.load(
                            projected_contribution_ptr + projected_base,
                            mask=mask_b[:, None] & (d_off[None, :] < D),
                            other=0.0,
                        )
                    else:
                        previous_xhat_base = (
                            ((layer - 1) * B + b64[:, None]) * T + t
                        ) * H + block * D
                        for k_iter in tl.static_range(tl.cdiv(D_PAD, BLOCK_K)):
                            kk = k_iter * BLOCK_K + k_off
                            previous_xhat = tl.load(
                                dynamic_normalized_ptr
                                + previous_xhat_base
                                + kk[None, :],
                                mask=mask_b[:, None] & (kk[None, :] < D),
                                other=0.0,
                            )
                            scale = tl.load(
                                dynamic_norm_ptr + (layer - 1) * H + block * D + kk,
                                mask=kk < D,
                                other=0.0,
                            )
                            previous = _activate(
                                previous_xhat * scale[None, :], ACTIVATION
                            )
                            weight = tl.load(
                                later_weight_ptr
                                + ((layer - 1) * NUM_BLOCKS + block) * D_PAD * D_PAD
                                + kk[:, None] * D_PAD
                                + d_off[None, :],
                                mask=(kk[:, None] < D_PAD) & (d_off[None, :] < D_PAD),
                                other=0.0,
                            )
                            if COMPUTE_BF16:
                                previous = previous.to(tl.bfloat16)
                            pre += tl.dot(previous, weight, input_precision="ieee")
                        bias = tl.load(
                            dynamic_bias_ptr + layer * H + block * D + d_off,
                            mask=d_off < D,
                            other=0.0,
                        )
                        pre += bias[None, :]
                    pre = tl.where(d_off[None, :] < D, pre, 0.0)
                    sumsq += tl.sum(pre * pre, axis=1)
                    scratch_base = (
                        ((layer * B + b64[:, None]) * T + t) * H
                        + block * D
                        + d_off[None, :]
                    )
                    tl.store(
                        dynamic_normalized_ptr + scratch_base,
                        pre,
                        mask=mask_b[:, None] & (d_off[None, :] < D),
                    )

                inv = tl.rsqrt(sumsq / H + NORM_EPS)
                if SAVE_STATE:
                    tl.store(
                        dynamic_inv_rms_ptr + (layer * B + b64) * T + t,
                        inv,
                        mask=mask_b,
                    )
                for block in tl.static_range(NUM_BLOCKS):
                    base = (
                        ((layer * B + b64[:, None]) * T + t) * H
                        + block * D
                        + d_off[None, :]
                    )
                    pre = tl.load(
                        dynamic_normalized_ptr + base,
                        mask=mask_b[:, None] & (d_off[None, :] < D),
                        other=0.0,
                    )
                    xhat = pre * inv[:, None]
                    tl.store(
                        dynamic_normalized_ptr + base,
                        xhat,
                        mask=mask_b[:, None] & (d_off[None, :] < D),
                    )

            for block in tl.static_range(NUM_BLOCKS):
                last_base = (
                    ((NUM_LAYERS - 1) * B + b64[:, None]) * T + t
                ) * H + block * D
                gate_pre = tl.zeros([BLOCK_B, G_PAD], tl.float32)
                gate_off = tl.arange(0, G_PAD)
                for k_iter in tl.static_range(tl.cdiv(D_PAD, BLOCK_K)):
                    kk = k_iter * BLOCK_K + k_off
                    xhat = tl.load(
                        dynamic_normalized_ptr + last_base + kk[None, :],
                        mask=mask_b[:, None] & (kk[None, :] < D),
                        other=0.0,
                    )
                    scale = tl.load(
                        dynamic_norm_ptr + (NUM_LAYERS - 1) * H + block * D + kk,
                        mask=kk < D,
                        other=0.0,
                    )
                    layer_value = _activate(xhat * scale[None, :], ACTIVATION)
                    weight = tl.load(
                        gate_weight_ptr
                        + block * D_PAD * G_PAD
                        + kk[:, None] * G_PAD
                        + gate_off[None, :],
                        mask=(kk[:, None] < D_PAD) & (gate_off[None, :] < G_PAD),
                        other=0.0,
                    )
                    if COMPUTE_BF16:
                        layer_value = layer_value.to(tl.bfloat16)
                    gate_pre += tl.dot(layer_value, weight, input_precision="ieee")
                for gate in tl.static_range(3):
                    bias = tl.load(
                        gate_bias_ptr + block * 3 * D + gate * D + d_off,
                        mask=d_off < D,
                        other=0.0,
                    )
                    gate_slice = gate_pre[:, gate * D_PAD + d_off] + bias[None, :]
                    if gate == 0:
                        reset_gate = tl.sigmoid(gate_slice)
                    elif gate == 1:
                        candidate_pre = gate_slice
                    else:
                        update = tl.sigmoid(gate_slice + UPDATE_BIAS)
                candidate = tl.extra.cuda.libdevice.tanh(reset_gate * candidate_pre)
                h_index = block * D + d_off
                previous_block = previous[:, h_index]
                next_block = update * candidate + (1.0 - update) * previous_block
                output_base = (b64[:, None] * T + t) * H + h_index[None, :]
                tl.store(
                    output_ptr + output_base,
                    next_block,
                    mask=mask_b[:, None] & (d_off[None, :] < D),
                )
                if SAVE_STATE:
                    state_base = (b64[:, None] * T + t) * (4 * H)
                    for state_index in tl.static_range(4):
                        if state_index == 0:
                            state = reset_gate
                        elif state_index == 1:
                            state = candidate_pre
                        elif state_index == 2:
                            state = candidate
                        else:
                            state = update
                        tl.store(
                            gate_state_ptr
                            + state_base
                            + state_index * H
                            + h_index[None, :],
                            state,
                            mask=mask_b[:, None] & (d_off[None, :] < D),
                        )
        final_value = tl.load(
            output_ptr + (b64[:, None] * T + T - 1) * H + h_off[None, :],
            mask=mask_b[:, None] & (h_off[None, :] < H),
            other=0.0,
        )
        tl.store(
            final_ptr + b64[:, None] * H + h_off[None, :],
            final_value,
            mask=mask_b[:, None] & (h_off[None, :] < H),
        )

    @triton.autotune(
        configs=_CONFIGS,
        key=[
            "B",
            "T",
            "H",
            "P",
            "NUM_BLOCKS",
            "NUM_LAYERS",
            "ACTIVATION",
            "COMPUTE_BF16",
        ],
        prune_configs_by={
            "perf_model": None,
            "top_k": None,
            "early_config_prune": _prune_configs,
        },
    )
    @triton.jit
    def _block_gru_bwd_kernel(
        initial_ptr,
        is_init_ptr,
        output_ptr,
        hidden_weight_t_ptr,
        hidden_norm_ptr,
        first_recurrent_weight_t_ptr,
        later_weight_t_ptr,
        dynamic_norm_ptr,
        gate_weight_t_ptr,
        hidden_normalized_ptr,
        hidden_inv_rms_ptr,
        dynamic_normalized_ptr,
        dynamic_inv_rms_ptr,
        gate_state_ptr,
        grad_output_ptr,
        grad_final_ptr,
        grad_initial_ptr,
        grad_hidden_pre_ptr,
        grad_hidden_norm_ptr,
        grad_dynamic_pre_ptr,
        grad_dynamic_norm_ptr,
        grad_gate_ptr,
        grad_layer_ptr,
        B,
        T,
        H: tl.constexpr,
        P: tl.constexpr,
        H_PAD: tl.constexpr,
        P_PAD: tl.constexpr,
        D: tl.constexpr,
        D_PAD: tl.constexpr,
        G_PAD: tl.constexpr,
        K_REC_PAD: tl.constexpr,
        NUM_BLOCKS: tl.constexpr,
        NUM_LAYERS: tl.constexpr,
        ACTIVATION: tl.constexpr,
        COMPUTE_BF16: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid * BLOCK_B + tl.arange(0, BLOCK_B)
        b64 = b.to(tl.int64)
        mask_b = b < B
        h_off = tl.arange(0, H_PAD)
        p_off = tl.arange(0, P_PAD)
        d_off = tl.arange(0, D_PAD)
        k_off = tl.arange(0, BLOCK_K)
        dh_next = tl.load(
            grad_final_ptr + b64[:, None] * H + h_off[None, :],
            mask=mask_b[:, None] & (h_off[None, :] < H),
            other=0.0,
        )

        for t_inv in range(T):
            t = T - 1 - t_inv
            base_h = (b64[:, None] * T + t) * H + h_off[None, :]
            dh = dh_next + tl.load(
                grad_output_ptr + base_h,
                mask=mask_b[:, None] & (h_off[None, :] < H),
                other=0.0,
            )
            reset_t = tl.load(is_init_ptr + b64 * T + t, mask=mask_b, other=False) != 0
            if t == 0:
                previous = tl.load(
                    initial_ptr + b64[:, None] * H + h_off[None, :],
                    mask=mask_b[:, None] & (h_off[None, :] < H),
                    other=0.0,
                )
            else:
                previous = tl.load(
                    output_ptr + (b64[:, None] * T + t - 1) * H + h_off[None, :],
                    mask=mask_b[:, None] & (h_off[None, :] < H),
                    other=0.0,
                )
            previous = tl.where(reset_t[:, None], 0.0, previous)
            update_full = tl.load(
                gate_state_ptr
                + (b64[:, None] * T + t) * (4 * H)
                + 3 * H
                + h_off[None, :],
                mask=mask_b[:, None] & (h_off[None, :] < H),
                other=0.0,
            )
            dh_previous = dh * (1.0 - update_full)

            for block in tl.static_range(NUM_BLOCKS):
                index = block * D + d_off
                state_base = (b64[:, None] * T + t) * (4 * H) + index[None, :]
                reset_gate = tl.load(
                    gate_state_ptr + state_base,
                    mask=mask_b[:, None] & (d_off[None, :] < D),
                    other=0.0,
                )
                candidate_pre = tl.load(
                    gate_state_ptr + state_base + H,
                    mask=mask_b[:, None] & (d_off[None, :] < D),
                    other=0.0,
                )
                candidate = tl.load(
                    gate_state_ptr + state_base + 2 * H,
                    mask=mask_b[:, None] & (d_off[None, :] < D),
                    other=0.0,
                )
                update = tl.load(
                    gate_state_ptr + state_base + 3 * H,
                    mask=mask_b[:, None] & (d_off[None, :] < D),
                    other=0.0,
                )
                dh_block = dh[:, index]
                previous_block = previous[:, index]
                grad_update_pre = (
                    dh_block * (candidate - previous_block) * update * (1.0 - update)
                )
                grad_candidate_inner = dh_block * update * (1.0 - candidate * candidate)
                grad_reset_pre = (
                    grad_candidate_inner
                    * candidate_pre
                    * reset_gate
                    * (1.0 - reset_gate)
                )
                grad_candidate_pre = grad_candidate_inner * reset_gate
                for gate in tl.static_range(3):
                    if gate == 0:
                        grad_gate_slice = grad_reset_pre
                    elif gate == 1:
                        grad_gate_slice = grad_candidate_pre
                    else:
                        grad_gate_slice = grad_update_pre
                    tl.store(
                        grad_gate_ptr
                        + (b64[:, None] * T + t) * (3 * H)
                        + block * 3 * D
                        + gate * D
                        + d_off[None, :],
                        grad_gate_slice,
                        mask=mask_b[:, None] & (d_off[None, :] < D),
                    )
                grad_value = tl.zeros([BLOCK_B, D_PAD], tl.float32)
                for gate in tl.static_range(3):
                    if gate == 0:
                        grad_gate_slice = grad_reset_pre
                    elif gate == 1:
                        grad_gate_slice = grad_candidate_pre
                    else:
                        grad_gate_slice = grad_update_pre
                    for k_iter in tl.static_range(tl.cdiv(D_PAD, BLOCK_K)):
                        kk = k_iter * BLOCK_K + k_off
                        gate_chunk = grad_gate_slice[:, kk]
                        weight_t = tl.load(
                            gate_weight_t_ptr
                            + block * G_PAD * D_PAD
                            + (gate * D_PAD + kk)[:, None] * D_PAD
                            + d_off[None, :],
                            mask=(kk[:, None] < D_PAD) & (d_off[None, :] < D_PAD),
                            other=0.0,
                        )
                        if COMPUTE_BF16:
                            gate_chunk = gate_chunk.to(tl.bfloat16)
                        grad_value += tl.dot(
                            gate_chunk, weight_t, input_precision="ieee"
                        )
                tl.store(
                    grad_layer_ptr + b64[:, None] * H + index[None, :],
                    grad_value,
                    mask=mask_b[:, None] & (d_off[None, :] < D),
                )

            hidden_feature_grad = tl.zeros([BLOCK_B, P_PAD], tl.float32)
            for layer_inv in tl.static_range(NUM_LAYERS):
                layer: tl.constexpr = NUM_LAYERS - 1 - layer_inv
                correction_sum = tl.zeros([BLOCK_B], tl.float32)
                inv = tl.load(
                    dynamic_inv_rms_ptr + (layer * B + b64) * T + t,
                    mask=mask_b,
                    other=0.0,
                )
                for block in tl.static_range(NUM_BLOCKS):
                    index = block * D + d_off
                    grad_value = tl.load(
                        grad_layer_ptr + b64[:, None] * H + index[None, :],
                        mask=mask_b[:, None] & (d_off[None, :] < D),
                        other=0.0,
                    )
                    base = ((layer * B + b64[:, None]) * T + t) * H + index[None, :]
                    xhat = tl.load(
                        dynamic_normalized_ptr + base,
                        mask=mask_b[:, None] & (d_off[None, :] < D),
                        other=0.0,
                    )
                    scale = tl.load(
                        dynamic_norm_ptr + layer * H + index,
                        mask=d_off < D,
                        other=0.0,
                    )
                    norm_value = xhat * scale[None, :]
                    grad_norm = grad_value * _activation_grad(norm_value, ACTIVATION)
                    tl.store(
                        grad_dynamic_norm_ptr + base,
                        grad_norm * xhat,
                        mask=mask_b[:, None] & (d_off[None, :] < D),
                    )
                    grad_scaled = grad_norm * scale[None, :]
                    tl.store(
                        grad_dynamic_pre_ptr + base,
                        grad_scaled,
                        mask=mask_b[:, None] & (d_off[None, :] < D),
                    )
                    correction_sum += tl.sum(grad_scaled * xhat, axis=1)
                correction = correction_sum / H

                for block in tl.static_range(NUM_BLOCKS):
                    index = block * D + d_off
                    base = ((layer * B + b64[:, None]) * T + t) * H + index[None, :]
                    xhat = tl.load(
                        dynamic_normalized_ptr + base,
                        mask=mask_b[:, None] & (d_off[None, :] < D),
                        other=0.0,
                    )
                    grad_scaled = tl.load(
                        grad_dynamic_pre_ptr + base,
                        mask=mask_b[:, None] & (d_off[None, :] < D),
                        other=0.0,
                    )
                    grad_pre = inv[:, None] * (grad_scaled - xhat * correction[:, None])
                    tl.store(
                        grad_dynamic_pre_ptr + base,
                        grad_pre,
                        mask=mask_b[:, None] & (d_off[None, :] < D),
                    )
                    if layer > 0:
                        grad_previous = tl.zeros([BLOCK_B, D_PAD], tl.float32)
                        for k_iter in tl.static_range(tl.cdiv(D_PAD, BLOCK_K)):
                            kk = k_iter * BLOCK_K + k_off
                            grad_chunk = grad_pre[:, kk]
                            weight_t = tl.load(
                                later_weight_t_ptr
                                + ((layer - 1) * NUM_BLOCKS + block) * D_PAD * D_PAD
                                + kk[:, None] * D_PAD
                                + d_off[None, :],
                                mask=(kk[:, None] < D_PAD) & (d_off[None, :] < D_PAD),
                                other=0.0,
                            )
                            if COMPUTE_BF16:
                                grad_chunk = grad_chunk.to(tl.bfloat16)
                            grad_previous += tl.dot(
                                grad_chunk, weight_t, input_precision="ieee"
                            )
                        tl.store(
                            grad_layer_ptr + b64[:, None] * H + index[None, :],
                            grad_previous,
                            mask=mask_b[:, None] & (d_off[None, :] < D),
                        )
                    else:
                        grad_recurrent = tl.zeros([BLOCK_B, K_REC_PAD], tl.float32)
                        rec_off = tl.arange(0, K_REC_PAD)
                        for k_iter in tl.static_range(tl.cdiv(D_PAD, BLOCK_K)):
                            kk = k_iter * BLOCK_K + k_off
                            grad_chunk = grad_pre[:, kk]
                            weight_t = tl.load(
                                first_recurrent_weight_t_ptr
                                + block * D_PAD * K_REC_PAD
                                + kk[:, None] * K_REC_PAD
                                + rec_off[None, :],
                                mask=(kk[:, None] < D_PAD)
                                & (rec_off[None, :] < K_REC_PAD),
                                other=0.0,
                            )
                            if COMPUTE_BF16:
                                grad_chunk = grad_chunk.to(tl.bfloat16)
                            grad_recurrent += tl.dot(
                                grad_chunk, weight_t, input_precision="ieee"
                            )
                        carry_grad = grad_recurrent[:, d_off]
                        tl.store(
                            grad_layer_ptr
                            + b64[:, None] * H
                            + block * D
                            + d_off[None, :],
                            carry_grad,
                            mask=mask_b[:, None] & (d_off[None, :] < D),
                        )
                        hidden_feature_grad += tl.where(
                            p_off[None, :] < P,
                            grad_recurrent[:, D + p_off],
                            0.0,
                        )

            carry_grad = tl.load(
                grad_layer_ptr + b64[:, None] * H + h_off[None, :],
                mask=mask_b[:, None] & (h_off[None, :] < H),
                other=0.0,
            )
            dh_previous += carry_grad
            hidden_base = (b64[:, None] * T + t) * P + p_off[None, :]
            hidden_xhat = tl.load(
                hidden_normalized_ptr + hidden_base,
                mask=mask_b[:, None] & (p_off[None, :] < P),
                other=0.0,
            )
            hidden_scale = tl.load(hidden_norm_ptr + p_off, mask=p_off < P, other=0.0)
            hidden_value = hidden_xhat * hidden_scale[None, :]
            hidden_grad_norm = hidden_feature_grad * _activation_grad(
                hidden_value, ACTIVATION
            )
            tl.store(
                grad_hidden_norm_ptr + hidden_base,
                hidden_grad_norm * hidden_xhat,
                mask=mask_b[:, None] & (p_off[None, :] < P),
            )
            hidden_grad_scaled = hidden_grad_norm * hidden_scale[None, :]
            hidden_correction = tl.sum(hidden_grad_scaled * hidden_xhat, axis=1) / P
            hidden_inv = tl.load(
                hidden_inv_rms_ptr + b64 * T + t, mask=mask_b, other=0.0
            )
            hidden_grad_pre = hidden_inv[:, None] * (
                hidden_grad_scaled - hidden_xhat * hidden_correction[:, None]
            )
            tl.store(
                grad_hidden_pre_ptr + hidden_base,
                hidden_grad_pre,
                mask=mask_b[:, None] & (p_off[None, :] < P),
            )
            hidden_carry_grad = tl.zeros([BLOCK_B, H_PAD], tl.float32)
            for k_iter in tl.static_range(tl.cdiv(P_PAD, BLOCK_K)):
                kk = k_iter * BLOCK_K + k_off
                grad_chunk = hidden_grad_pre[:, kk]
                weight_t = tl.load(
                    hidden_weight_t_ptr + kk[:, None] * H_PAD + h_off[None, :],
                    mask=(kk[:, None] < P_PAD) & (h_off[None, :] < H_PAD),
                    other=0.0,
                )
                if COMPUTE_BF16:
                    grad_chunk = grad_chunk.to(tl.bfloat16)
                hidden_carry_grad += tl.dot(
                    grad_chunk, weight_t, input_precision="ieee"
                )
            dh_previous += hidden_carry_grad
            dh_next = tl.where(reset_t[:, None], 0.0, dh_previous)

        tl.store(
            grad_initial_ptr + b64[:, None] * H + h_off[None, :],
            dh_next,
            mask=mask_b[:, None] & (h_off[None, :] < H),
        )


def _pad_matrix(value: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    return F.pad(value, (0, cols - value.shape[-1], 0, rows - value.shape[-2]))


class _DreamerV3BlockGRUTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        projected_input,
        initial_hidden,
        is_init,
        hidden_weight,
        hidden_bias,
        hidden_norm_weight,
        *args,
    ):
        if not _has_triton:
            raise RuntimeError(
                "The DreamerV3 block-GRU Triton backend requires Triton."
            )
        (
            activation_code,
            norm_eps,
            update_bias,
            num_blocks,
            num_layers,
            save_state,
        ) = args[-6:]
        tensors = args[:-6]
        dynamic = [tensors[index : index + 3] for index in range(0, 3 * num_layers, 3)]
        gate_weight, gate_bias = tensors[3 * num_layers :]
        batch, time, projected_size = projected_input.shape
        hidden_size = initial_hidden.shape[-1]
        block_size = hidden_size // num_blocks
        compute_dtype = projected_input.dtype
        h_pad = max(
            16,
            triton.next_power_of_2(
                num_blocks * max(16, triton.next_power_of_2(block_size))
            ),
        )
        p_pad = max(16, triton.next_power_of_2(projected_size))
        d_pad = max(16, triton.next_power_of_2(block_size))
        g_pad = 4 * d_pad
        k_rec_pad = max(16, triton.next_power_of_2(block_size + projected_size))

        hidden_weight_p = _pad_matrix(
            hidden_weight.to(compute_dtype).t().contiguous(), h_pad, p_pad
        ).contiguous()
        hidden_weight_t_p = hidden_weight_p.t().contiguous()
        hidden_bias_p = F.pad(
            hidden_bias.to(compute_dtype), (0, p_pad - projected_size)
        )
        hidden_norm_p = F.pad(hidden_norm_weight, (0, p_pad - projected_size))

        first_weight = dynamic[0][0].to(compute_dtype)
        first_carry = first_weight[:, :block_size]
        first_input = first_weight[:, block_size : block_size + projected_size]
        first_hidden = first_weight[:, block_size + projected_size :]
        first_recurrent = torch.cat((first_carry, first_hidden), 1)
        first_recurrent_p = F.pad(
            first_recurrent,
            (0, d_pad - block_size, 0, k_rec_pad - first_recurrent.shape[1]),
        ).contiguous()
        first_recurrent_t_p = first_recurrent_p.transpose(1, 2).contiguous()
        projected_contribution = torch.bmm(
            projected_input.flatten(0, 1).unsqueeze(0).expand(num_blocks, -1, -1),
            first_input,
        )
        projected_contribution = (
            projected_contribution.transpose(0, 1)
            .reshape(batch, time, hidden_size)
            .add(dynamic[0][1].to(compute_dtype))
            .contiguous()
        )

        later = []
        later_t = []
        for weight, _, _ in dynamic[1:]:
            weight_p = F.pad(
                weight.to(compute_dtype),
                (0, d_pad - block_size, 0, d_pad - block_size),
            )
            later.append(weight_p)
            later_t.append(weight_p.transpose(1, 2))
        later_weight_p = (
            torch.stack(later).contiguous()
            if later
            else projected_input.new_empty((0, num_blocks, d_pad, d_pad))
        )
        later_weight_t_p = (
            torch.stack(later_t).contiguous()
            if later_t
            else projected_input.new_empty((0, num_blocks, d_pad, d_pad))
        )
        dynamic_bias_p = torch.stack(
            [
                F.pad(bias.to(compute_dtype), (0, h_pad - hidden_size))
                for _, bias, _ in dynamic
            ]
        ).contiguous()
        dynamic_norm_p = torch.stack(
            [F.pad(norm, (0, h_pad - hidden_size)) for _, _, norm in dynamic]
        ).contiguous()
        gate_weight_p = projected_input.new_zeros(
            num_blocks, d_pad, g_pad, dtype=compute_dtype
        )
        for gate in range(3):
            gate_weight_p[
                :, :block_size, gate * d_pad : gate * d_pad + block_size
            ] = gate_weight.to(compute_dtype)[
                :, :, gate * block_size : (gate + 1) * block_size
            ]
        gate_weight_t_p = gate_weight_p.transpose(1, 2).contiguous()

        outputs = torch.empty(
            batch, time, hidden_size, device=projected_input.device, dtype=compute_dtype
        )
        final_hidden = torch.empty_like(initial_hidden)
        hidden_normalized = torch.empty(
            batch,
            time,
            projected_size,
            device=projected_input.device,
            dtype=compute_dtype,
        )
        hidden_inv_rms = torch.empty(
            batch, time, device=projected_input.device, dtype=torch.float32
        )
        dynamic_normalized = torch.empty(
            num_layers,
            batch,
            time,
            hidden_size,
            device=projected_input.device,
            dtype=compute_dtype,
        )
        dynamic_inv_rms = torch.empty(
            num_layers, batch, time, device=projected_input.device, dtype=torch.float32
        )
        gate_state = torch.empty(
            batch,
            time,
            4 * hidden_size,
            device=projected_input.device,
            dtype=compute_dtype,
        )

        def grid(meta):
            return (triton.cdiv(batch, meta["BLOCK_B"]),)

        _block_gru_fwd_kernel[grid](
            projected_contribution,
            initial_hidden,
            is_init,
            hidden_weight_p,
            hidden_bias_p,
            hidden_norm_p,
            first_recurrent_p,
            later_weight_p,
            dynamic_bias_p,
            dynamic_norm_p,
            gate_weight_p,
            gate_bias.to(compute_dtype).contiguous(),
            outputs,
            final_hidden,
            hidden_normalized,
            hidden_inv_rms,
            dynamic_normalized,
            dynamic_inv_rms,
            gate_state,
            batch,
            time,
            H=hidden_size,
            P=projected_size,
            H_PAD=h_pad,
            P_PAD=p_pad,
            D=block_size,
            D_PAD=d_pad,
            G_PAD=g_pad,
            K_REC_PAD=k_rec_pad,
            NUM_BLOCKS=num_blocks,
            NUM_LAYERS=num_layers,
            NORM_EPS=norm_eps,
            UPDATE_BIAS=update_bias,
            ACTIVATION=activation_code,
            COMPUTE_BF16=compute_dtype is torch.bfloat16,
            SAVE_STATE=save_state,
        )
        ctx.activation_code = activation_code
        ctx.num_blocks = num_blocks
        ctx.num_layers = num_layers
        ctx.padding = (h_pad, p_pad, d_pad, g_pad, k_rec_pad)
        ctx.save_for_backward(
            projected_input,
            initial_hidden,
            is_init,
            hidden_weight,
            hidden_norm_weight,
            *tensors,
            outputs,
            hidden_normalized,
            hidden_inv_rms,
            dynamic_normalized,
            dynamic_inv_rms,
            gate_state,
            hidden_weight_t_p,
            first_recurrent_t_p,
            later_weight_t_p,
            dynamic_norm_p,
            gate_weight_t_p,
        )
        return outputs, final_hidden

    @staticmethod
    def backward(ctx, grad_outputs, grad_final_hidden):
        saved = ctx.saved_tensors
        num_layers = ctx.num_layers
        parameter_count = 3 * num_layers + 2
        (
            projected_input,
            initial_hidden,
            is_init,
            hidden_weight,
            hidden_norm_weight,
        ) = saved[:5]
        parameters = saved[5 : 5 + parameter_count]
        dynamic = [
            parameters[index : index + 3] for index in range(0, 3 * num_layers, 3)
        ]
        gate_weight, _ = parameters[3 * num_layers :]
        offset = 5 + parameter_count
        (
            outputs,
            hidden_normalized,
            hidden_inv_rms,
            dynamic_normalized,
            dynamic_inv_rms,
            gate_state,
            hidden_weight_t_p,
            first_recurrent_t_p,
            later_weight_t_p,
            dynamic_norm_p,
            gate_weight_t_p,
        ) = saved[offset:]
        batch, time, projected_size = projected_input.shape
        hidden_size = initial_hidden.shape[-1]
        block_size = hidden_size // ctx.num_blocks
        h_pad, p_pad, d_pad, g_pad, k_rec_pad = ctx.padding
        if grad_outputs is None:
            grad_outputs = torch.zeros_like(outputs)
        if grad_final_hidden is None:
            grad_final_hidden = torch.zeros_like(initial_hidden)
        grad_outputs = grad_outputs.contiguous()
        grad_final_hidden = grad_final_hidden.contiguous()
        grad_initial = torch.empty_like(initial_hidden)
        grad_hidden_pre = torch.empty(
            batch, time, projected_size, device=outputs.device, dtype=outputs.dtype
        )
        grad_hidden_norm_contrib = torch.empty_like(grad_hidden_pre)
        grad_dynamic_pre = torch.empty_like(dynamic_normalized)
        grad_dynamic_norm_contrib = torch.empty_like(dynamic_normalized)
        grad_gate = torch.empty(
            batch, time, 3 * hidden_size, device=outputs.device, dtype=outputs.dtype
        )
        grad_layer = torch.empty(
            batch, hidden_size, device=outputs.device, dtype=outputs.dtype
        )
        hidden_norm_p = F.pad(
            hidden_norm_weight, (0, p_pad - projected_size)
        ).contiguous()

        def grid(meta):
            return (triton.cdiv(batch, meta["BLOCK_B"]),)

        _block_gru_bwd_kernel[grid](
            initial_hidden,
            is_init,
            outputs,
            hidden_weight_t_p,
            hidden_norm_p,
            first_recurrent_t_p,
            later_weight_t_p,
            dynamic_norm_p,
            gate_weight_t_p,
            hidden_normalized,
            hidden_inv_rms,
            dynamic_normalized,
            dynamic_inv_rms,
            gate_state,
            grad_outputs,
            grad_final_hidden,
            grad_initial,
            grad_hidden_pre,
            grad_hidden_norm_contrib,
            grad_dynamic_pre,
            grad_dynamic_norm_contrib,
            grad_gate,
            grad_layer,
            batch,
            time,
            H=hidden_size,
            P=projected_size,
            H_PAD=h_pad,
            P_PAD=p_pad,
            D=block_size,
            D_PAD=d_pad,
            G_PAD=g_pad,
            K_REC_PAD=k_rec_pad,
            NUM_BLOCKS=ctx.num_blocks,
            NUM_LAYERS=num_layers,
            ACTIVATION=ctx.activation_code,
            COMPUTE_BF16=outputs.dtype is torch.bfloat16,
        )

        previous = torch.cat((initial_hidden.unsqueeze(1), outputs[:, :-1]), 1)
        previous = torch.where(is_init.unsqueeze(-1), 0, previous)
        flat_previous = previous.flatten(0, 1).float()
        flat_hidden_pre = grad_hidden_pre.flatten(0, 1).float()
        grad_hidden_weight = flat_hidden_pre.t() @ flat_previous
        grad_hidden_bias = flat_hidden_pre.sum(0)
        grad_hidden_norm_weight = grad_hidden_norm_contrib.float().sum((0, 1))

        hidden_features = _activation(
            hidden_normalized * hidden_norm_weight.to(outputs.dtype),
            ctx.activation_code,
        )
        grouped_previous = previous.unflatten(-1, (ctx.num_blocks, block_size))
        repeated_projected = projected_input.unsqueeze(-2).expand(
            batch, time, ctx.num_blocks, projected_size
        )
        repeated_hidden = hidden_features.unsqueeze(-2).expand_as(repeated_projected)
        first_input = torch.cat(
            (grouped_previous, repeated_projected, repeated_hidden), -1
        ).flatten(-2)
        dynamic_grads = []
        for layer_index, (weight, _, _) in enumerate(dynamic):
            grad_pre = grad_dynamic_pre[layer_index]
            layer_input = (
                first_input
                if layer_index == 0
                else _activation(
                    dynamic_normalized[layer_index - 1]
                    * dynamic[layer_index - 1][2].to(outputs.dtype),
                    ctx.activation_code,
                )
            )
            dynamic_grads.extend(
                (
                    _block_weight_grad(layer_input, grad_pre, weight),
                    grad_pre.float().sum((0, 1)),
                    grad_dynamic_norm_contrib[layer_index].float().sum((0, 1)),
                )
            )
        first_grad_blocks = (
            grad_dynamic_pre[0].reshape(-1, ctx.num_blocks, block_size).transpose(0, 1)
        )
        first_weight = dynamic[0][0].to(outputs.dtype)
        input_weight = first_weight[:, block_size : block_size + projected_size]
        grad_projected = (
            torch.bmm(first_grad_blocks, input_weight.transpose(1, 2))
            .sum(0)
            .reshape_as(projected_input)
        )
        gate_input = _activation(
            dynamic_normalized[-1] * dynamic[-1][2].to(outputs.dtype),
            ctx.activation_code,
        )
        grad_gate_weight = _block_weight_grad(gate_input, grad_gate, gate_weight)
        grad_gate_bias = grad_gate.float().sum((0, 1))
        tensor_grads = (
            grad_projected.to(projected_input.dtype),
            grad_initial.to(initial_hidden.dtype),
            None,
            grad_hidden_weight,
            grad_hidden_bias,
            grad_hidden_norm_weight,
            *dynamic_grads,
            grad_gate_weight,
            grad_gate_bias,
        )
        return (*tensor_grads, None, None, None, None, None, None)


def dreamer_v3_block_gru_triton(
    projected_input: torch.Tensor,
    initial_hidden: torch.Tensor,
    is_init: torch.Tensor,
    hidden_weight: torch.Tensor,
    hidden_bias: torch.Tensor,
    hidden_norm_weight: torch.Tensor,
    dynamic_parameters: list[torch.Tensor],
    gate_weight: torch.Tensor,
    gate_bias: torch.Tensor,
    activation: nn.Module,
    norm_eps: float,
    update_bias: float,
    num_blocks: int,
    num_layers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the CUDA-only fused DreamerV3 block-GRU recurrence."""
    if not _has_triton:
        raise RuntimeError("recurrent_backend='triton' requires Triton 3.3 or newer.")
    if not projected_input.is_cuda:
        raise RuntimeError("recurrent_backend='triton' requires CUDA tensors.")
    activation_code = _activation_code(activation)
    save_state = torch.is_grad_enabled() and any(
        tensor.requires_grad
        for tensor in (
            projected_input,
            initial_hidden,
            hidden_weight,
            hidden_bias,
            hidden_norm_weight,
            *dynamic_parameters,
            gate_weight,
            gate_bias,
        )
    )
    return _DreamerV3BlockGRUTritonFunction.apply(
        projected_input.contiguous(),
        initial_hidden.contiguous(),
        is_init.contiguous(),
        hidden_weight,
        hidden_bias,
        hidden_norm_weight,
        *dynamic_parameters,
        gate_weight,
        gate_bias,
        activation_code,
        norm_eps,
        update_bias,
        num_blocks,
        num_layers,
        save_state,
    )
