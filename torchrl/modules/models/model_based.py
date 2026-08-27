# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Model-based reinforcement-learning components.

Includes the continuous Dreamer RSSM and the discrete DreamerV3 RSSM.
"""
from __future__ import annotations

import functools as ft
import warnings

from collections.abc import Callable
from typing import Literal

import torch
from packaging import version
from tensordict.nn import (
    NormalParamExtractor,
    TensorDictModule,
    TensorDictModuleBase,
    TensorDictSequential,
)
from tensordict.utils import NestedKey, unravel_key
from torch import nn
from torch.autograd.function import once_differentiable
from torch.nn import functional as F, GRUCell
from torchrl._utils import implement_for
from torchrl.modules.functional import symexp, symlog  # noqa: F401
from torchrl.modules.models.models import MLP
from torchrl.modules.tensordict_module.rnn import (
    _maybe_warm_scan_backward,
    _scan as _higher_order_scan,
)


_DEFAULT_NUM_BINS = 255
_DEFAULT_BIN_RANGE = 20.0
UNSQUEEZE_RNN_INPUT = version.parse(torch.__version__) < version.parse("1.11")


def _dreamer_v3_init(module: nn.Module) -> None:
    """Initialize linear modules like the reference DreamerV3 implementation."""
    if isinstance(module, nn.Linear):
        std = 1.1368 / module.in_features**0.5
        nn.init.trunc_normal_(module.weight, std=std, a=-2 * std, b=2 * std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class _DreamerV3RMSNorm(nn.Module):
    """RMS normalization with a learned scale and no shift."""

    def __init__(self, features: int, eps: float = 1e-4, device=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(features, device=device))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return _dreamer_v3_rms_norm(value, self.weight, self.eps)


class _DreamerV3BlockLinear(nn.Module):
    """Independent linear projections over equally sized feature blocks."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_blocks: int,
        *,
        device=None,
    ):
        super().__init__()
        if in_features % num_blocks or out_features % num_blocks:
            raise ValueError(
                "in_features and out_features must be divisible by num_blocks, "
                f"got {in_features}, {out_features}, and {num_blocks}."
            )
        self.in_features = in_features
        self.out_features = out_features
        self.num_blocks = num_blocks
        block_in = in_features // num_blocks
        block_out = out_features // num_blocks
        self.weight = nn.Parameter(
            torch.empty(num_blocks, block_in, block_out, device=device)
        )
        self.bias = nn.Parameter(torch.zeros(out_features, device=device))
        # Fan-in spans the whole kernel: divide by in_features, not block_in.
        std = 1.1368 / in_features**0.5
        nn.init.trunc_normal_(self.weight, std=std, a=-2 * std, b=2 * std)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch_shape = value.shape[:-1]
        value = value.reshape(
            -1, self.num_blocks, self.in_features // self.num_blocks
        ).transpose(0, 1)
        value = torch.bmm(value, self.weight.to(value.dtype)).transpose(0, 1)
        # An FP32 bias would promote a BF16 recurrence back to FP32.
        return value.reshape(*batch_shape, self.out_features) + self.bias.to(
            value.dtype
        )


def _dreamer_v3_linear(
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Run a linear projection in the activation dtype with FP32 parameters."""
    dtype = value.dtype
    return F.linear(
        value,
        weight.to(dtype),
        bias.to(dtype) if bias is not None else None,
    )


@implement_for("torch", None, "2.4", compilable=True)
def _dreamer_v3_rms_norm(
    value: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    dtype = value.dtype
    value = value.float()
    value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
    return (value * weight.float()).to(dtype)


@implement_for("torch", "2.4", compilable=True)
def _dreamer_v3_rms_norm(  # noqa: F811
    value: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return F.rms_norm(value.float(), (weight.shape[0],), weight.float(), eps).to(
        value.dtype
    )


def _dreamer_v3_block_linear(
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    num_blocks, _, block_out = weight.shape
    batch_shape = value.shape[:-1]
    value = value.reshape(-1, num_blocks, value.shape[-1] // num_blocks)
    value = value.transpose(0, 1)
    value = torch.bmm(value, weight.to(value.dtype)).transpose(0, 1)
    return value.reshape(*batch_shape, num_blocks * block_out) + bias.to(value.dtype)


def _dreamer_v3_block_gru_update(
    features: torch.Tensor,
    belief: torch.Tensor,
    hidden_layers: Callable[[torch.Tensor], torch.Tensor],
    gates: _DreamerV3BlockLinear,
    *,
    num_blocks: int,
    update_bias: float,
) -> torch.Tensor:
    """Apply the shared DreamerV3 block dynamics and gated update."""
    belief_dim = belief.shape[-1]
    grouped_belief = belief.reshape(
        *belief.shape[:-1], num_blocks, belief_dim // num_blocks
    )
    repeated_features = features.unsqueeze(-2).expand(
        *features.shape[:-1], num_blocks, features.shape[-1]
    )
    hidden = torch.cat([grouped_belief, repeated_features], -1).flatten(-2)
    hidden = hidden_layers(hidden)
    gate_values = gates(hidden).reshape(
        *hidden.shape[:-1], num_blocks, 3, belief_dim // num_blocks
    )
    reset, candidate, update = gate_values.unbind(-2)
    reset = reset.flatten(-2).sigmoid()
    candidate = (reset * candidate.flatten(-2)).tanh()
    update = (update.flatten(-2) + update_bias).sigmoid()
    return update * candidate + (1 - update) * belief


class _DreamerV3BlockGRU(nn.Module):
    """Grouped gated recurrent core used by the reference DreamerV3 RSSM."""

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        belief_dim: int,
        num_blocks: int,
        num_layers: int,
        norm_eps: float,
        device=None,
    ):
        super().__init__()
        if belief_dim % num_blocks:
            raise ValueError(
                "rnn_hidden_dim must be divisible by num_blocks, got "
                f"{belief_dim} and {num_blocks}."
            )
        self.belief_dim = belief_dim
        self.num_blocks = num_blocks
        self.belief_projection = nn.Sequential(
            nn.Linear(belief_dim, hidden_dim, device=device),
            _DreamerV3RMSNorm(hidden_dim, norm_eps, device=device),
            nn.SiLU(),
        )
        self.state_projection = nn.Sequential(
            nn.Linear(state_dim, hidden_dim, device=device),
            _DreamerV3RMSNorm(hidden_dim, norm_eps, device=device),
            nn.SiLU(),
        )
        self.action_projection = nn.Sequential(
            nn.Linear(action_dim, hidden_dim, device=device),
            _DreamerV3RMSNorm(hidden_dim, norm_eps, device=device),
            nn.SiLU(),
        )
        layer_in = belief_dim + 3 * hidden_dim * num_blocks
        layers = []
        for _ in range(num_layers):
            layers.extend(
                [
                    _DreamerV3BlockLinear(
                        layer_in, belief_dim, num_blocks, device=device
                    ),
                    _DreamerV3RMSNorm(belief_dim, norm_eps, device=device),
                    nn.SiLU(),
                ]
            )
            layer_in = belief_dim
        self.hidden_layers = nn.Sequential(*layers)
        self.gates = _DreamerV3BlockLinear(
            belief_dim, 3 * belief_dim, num_blocks, device=device
        )
        self.apply(_dreamer_v3_init)

    def forward(
        self,
        state: torch.Tensor,
        belief: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        action = action / action.detach().abs().clamp_min(1)
        features = torch.cat(
            [
                self.belief_projection(belief),
                self.state_projection(state),
                self.action_projection(action),
            ],
            -1,
        )
        return _dreamer_v3_block_gru_update(
            features,
            belief,
            self.hidden_layers,
            self.gates,
            num_blocks=self.num_blocks,
            update_bias=-1.0,
        )


def _activation_with_derivative(
    activation: Callable[[torch.Tensor], torch.Tensor],
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(activation, nn.SiLU):
        sigmoid = value.sigmoid()
        output = F.silu(value)
        return output, sigmoid * (1 + value * (1 - sigmoid))
    if isinstance(activation, nn.Tanh):
        output = value.tanh()
        return output, 1 - output.square()
    if isinstance(activation, nn.ReLU):
        return value.relu(), (value > 0).to(value.dtype)
    output, derivative = torch.func.jvp(activation, (value,), (torch.ones_like(value),))
    return output, derivative


def _rms_norm_with_backward_state(
    value: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    value_float = value.float()
    inv_rms = torch.rsqrt(value_float.square().mean(-1, keepdim=True) + eps)
    normalized = value_float * inv_rms
    output = _dreamer_v3_rms_norm(value, weight, eps)
    return output, normalized.to(value.dtype), inv_rms


def _rms_norm_backward(
    grad_output: torch.Tensor,
    normalized: torch.Tensor,
    inv_rms: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    grad_float = grad_output.float()
    normalized_float = normalized.float()
    grad_normalized = grad_float * weight.float()
    correction = (grad_normalized * normalized_float).mean(-1, keepdim=True)
    grad_input = inv_rms * (grad_normalized - normalized_float * correction)
    grad_weight_contribution = grad_float * normalized_float
    return grad_input.to(grad_output.dtype), grad_weight_contribution


def _block_linear_backward_input(
    grad_output: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    num_blocks, block_in, block_out = weight.shape
    batch_shape = grad_output.shape[:-1]
    grad_blocks = grad_output.reshape(-1, num_blocks, block_out)
    grad_blocks = grad_blocks.transpose(0, 1)
    grad_input = torch.bmm(grad_blocks, weight.to(grad_output.dtype).transpose(1, 2))
    return grad_input.transpose(0, 1).reshape(*batch_shape, num_blocks * block_in)


def _block_weight_grad(
    value: torch.Tensor,
    grad_output: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    num_blocks, block_in, block_out = weight.shape
    value = value.reshape(-1, num_blocks, block_in).float()
    grad_output = grad_output.reshape(-1, num_blocks, block_out).float()
    return torch.einsum("nbi,nbo->bio", value, grad_output)


class _DreamerV3BlockGRUScanFunction(torch.autograd.Function):
    """Block-GRU scan whose reverse scan carries only the hidden cotangent."""

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
        activation, norm_eps, update_bias, num_blocks, num_layers = args[-5:]
        tensors = args[:-5]
        dynamic = [tensors[index : index + 3] for index in range(0, 3 * num_layers, 3)]
        gate_weight, gate_bias = tensors[3 * num_layers :]
        hidden_size = initial_hidden.shape[-1]
        block_size = hidden_size // num_blocks

        def step(hidden, inputs):
            projected_t, init_t = inputs
            previous = torch.where(init_t.unsqueeze(-1), 0, hidden)

            hidden_pre = _dreamer_v3_linear(previous, hidden_weight, hidden_bias)
            (
                hidden_norm,
                hidden_normalized,
                hidden_inv_rms,
            ) = _rms_norm_with_backward_state(hidden_pre, hidden_norm_weight, norm_eps)
            hidden_features, hidden_activation_derivative = _activation_with_derivative(
                activation, hidden_norm
            )
            features = torch.cat((projected_t, hidden_features), -1)
            grouped_hidden = previous.reshape(previous.shape[0], num_blocks, block_size)
            repeated_features = features.unsqueeze(-2).expand(
                features.shape[0], num_blocks, features.shape[-1]
            )
            layer_value = torch.cat((grouped_hidden, repeated_features), -1).flatten(-2)

            layer_inputs = []
            layer_normalized = []
            layer_inv_rms = []
            layer_derivatives = []
            for weight, bias, norm_weight in dynamic:
                layer_inputs.append(layer_value)
                layer_pre = _dreamer_v3_block_linear(layer_value, weight, bias)
                layer_norm, normalized, inv_rms = _rms_norm_with_backward_state(
                    layer_pre, norm_weight, norm_eps
                )
                layer_value, derivative = _activation_with_derivative(
                    activation, layer_norm
                )
                layer_normalized.append(normalized)
                layer_inv_rms.append(inv_rms)
                layer_derivatives.append(derivative)

            gate_values = _dreamer_v3_block_linear(
                layer_value, gate_weight, gate_bias
            ).reshape(previous.shape[0], num_blocks, 3, block_size)
            reset_pre, candidate_pre, update_pre = gate_values.unbind(-2)
            reset = reset_pre.flatten(-2).sigmoid()
            candidate_pre = candidate_pre.flatten(-2)
            candidate = (reset * candidate_pre).tanh()
            update = (update_pre.flatten(-2) + update_bias).sigmoid()
            next_hidden = update * candidate + (1 - update) * previous

            saved = (
                previous,
                hidden_normalized,
                hidden_inv_rms,
                hidden_activation_derivative,
                *layer_inputs,
                *layer_normalized,
                *layer_inv_rms,
                *layer_derivatives,
                layer_value,
                reset,
                candidate_pre,
                candidate,
                update,
            )
            return next_hidden, (next_hidden.clone(), *saved)

        final_hidden, scan_output = _higher_order_scan(
            step,
            initial_hidden,
            (projected_input, is_init),
            dim=0,
        )
        outputs, saved = scan_output[0], scan_output[1:]
        ctx.activation = activation
        ctx.norm_eps = norm_eps
        ctx.update_bias = update_bias
        ctx.num_blocks = num_blocks
        ctx.num_layers = num_layers
        ctx.save_for_backward(
            projected_input,
            initial_hidden,
            is_init,
            hidden_weight,
            hidden_norm_weight,
            *tensors,
            outputs,
            *saved,
        )
        return outputs, final_hidden

    # The saved gate states carry no autograd history, so double backward
    # would silently return wrong second-order gradients without this.
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_outputs, grad_final_hidden):
        saved_tensors = ctx.saved_tensors
        num_layers = ctx.num_layers
        parameter_count = 3 * num_layers + 2
        (
            projected_input,
            initial_hidden,
            is_init,
            hidden_weight,
            hidden_norm_weight,
        ) = saved_tensors[:5]
        parameter_tensors = saved_tensors[5 : 5 + parameter_count]
        dynamic = [
            parameter_tensors[index : index + 3]
            for index in range(0, 3 * num_layers, 3)
        ]
        gate_weight, _ = parameter_tensors[3 * num_layers :]
        outputs = saved_tensors[5 + parameter_count]
        saved = saved_tensors[6 + parameter_count :]

        offset = 0
        previous = saved[offset]
        offset += 1
        hidden_normalized = saved[offset]
        hidden_inv_rms = saved[offset + 1]
        hidden_derivative = saved[offset + 2]
        offset += 3
        layer_inputs = saved[offset : offset + num_layers]
        offset += num_layers
        layer_normalized = saved[offset : offset + num_layers]
        offset += num_layers
        layer_inv_rms = saved[offset : offset + num_layers]
        offset += num_layers
        layer_derivatives = saved[offset : offset + num_layers]
        offset += num_layers
        gate_input, reset, candidate_pre, candidate, update = saved[offset : offset + 5]

        if grad_outputs is None:
            grad_outputs = torch.zeros_like(outputs)
        if grad_final_hidden is None:
            grad_final_hidden = torch.zeros_like(initial_hidden)
        grad_outputs = torch.cat(
            (grad_outputs[:-1], (grad_outputs[-1] + grad_final_hidden).unsqueeze(0)),
            0,
        )
        num_blocks = ctx.num_blocks
        hidden_size = initial_hidden.shape[-1]
        block_size = hidden_size // num_blocks
        projected_size = projected_input.shape[-1]

        reversed_inputs = tuple(
            value.flip(0)
            for value in (
                grad_outputs,
                is_init,
                previous,
                hidden_normalized,
                hidden_inv_rms,
                hidden_derivative,
                *layer_normalized,
                *layer_inv_rms,
                *layer_derivatives,
                gate_input,
                reset,
                candidate_pre,
                candidate,
                update,
            )
        )

        def reverse_step(hidden_cotangent, inputs):
            index = 0
            output_cotangent = inputs[index]
            init_t = inputs[index + 1]
            previous_t = inputs[index + 2]
            hidden_normalized_t = inputs[index + 3]
            hidden_inv_rms_t = inputs[index + 4]
            hidden_derivative_t = inputs[index + 5]
            index += 6
            layer_normalized_t = inputs[index : index + num_layers]
            index += num_layers
            layer_inv_rms_t = inputs[index : index + num_layers]
            index += num_layers
            layer_derivatives_t = inputs[index : index + num_layers]
            index += num_layers
            gate_input_t, reset_t, candidate_pre_t, candidate_t, update_t = inputs[
                index : index + 5
            ]

            grad_hidden = output_cotangent + hidden_cotangent
            grad_update = grad_hidden * (candidate_t - previous_t)
            grad_candidate = grad_hidden * update_t
            grad_previous = grad_hidden * (1 - update_t)
            grad_update_pre = grad_update * update_t * (1 - update_t)
            grad_candidate_inner = grad_candidate * (1 - candidate_t.square())
            grad_reset = grad_candidate_inner * candidate_pre_t
            grad_candidate_pre = grad_candidate_inner * reset_t
            grad_reset_pre = grad_reset * reset_t * (1 - reset_t)
            grad_gates = torch.stack(
                (
                    grad_reset_pre.unflatten(-1, (num_blocks, block_size)),
                    grad_candidate_pre.unflatten(-1, (num_blocks, block_size)),
                    grad_update_pre.unflatten(-1, (num_blocks, block_size)),
                ),
                -2,
            ).flatten(-3)

            grad_layer = _block_linear_backward_input(grad_gates, gate_weight)
            dynamic_pre_grads = []
            dynamic_norm_grads = []
            for layer_index in range(num_layers - 1, -1, -1):
                weight, _, norm_weight = dynamic[layer_index]
                grad_norm = grad_layer * layer_derivatives_t[layer_index]
                grad_pre, grad_norm_weight = _rms_norm_backward(
                    grad_norm,
                    layer_normalized_t[layer_index],
                    layer_inv_rms_t[layer_index],
                    norm_weight,
                )
                dynamic_pre_grads.append(grad_pre)
                dynamic_norm_grads.append(grad_norm_weight)
                grad_layer = _block_linear_backward_input(grad_pre, weight)
            dynamic_pre_grads.reverse()
            dynamic_norm_grads.reverse()

            grad_first = grad_layer.reshape(grad_layer.shape[0], num_blocks, -1)
            grad_previous = grad_previous + grad_first[..., :block_size].flatten(-2)
            grad_features = grad_first[..., block_size:].sum(-2)
            grad_projected = grad_features[..., :projected_size]
            grad_hidden_features = grad_features[..., projected_size:]

            grad_hidden_norm = grad_hidden_features * hidden_derivative_t
            grad_hidden_pre, grad_hidden_norm_weight = _rms_norm_backward(
                grad_hidden_norm,
                hidden_normalized_t,
                hidden_inv_rms_t,
                hidden_norm_weight,
            )
            grad_previous = grad_previous + _dreamer_v3_linear(
                grad_hidden_pre, hidden_weight.t(), None
            )
            next_cotangent = torch.where(init_t.unsqueeze(-1), 0, grad_previous)
            local = (
                grad_projected,
                grad_hidden_pre,
                grad_hidden_norm_weight,
                *dynamic_pre_grads,
                *dynamic_norm_grads,
                grad_gates,
            )
            return next_cotangent, local

        grad_initial, local = _higher_order_scan(
            reverse_step,
            torch.zeros_like(initial_hidden),
            reversed_inputs,
            dim=0,
        )
        local = tuple(value.flip(0) for value in local)
        index = 0
        grad_projected = local[index]
        grad_hidden_pre = local[index + 1]
        grad_hidden_norm_contrib = local[index + 2]
        index += 3
        dynamic_pre_grads = local[index : index + num_layers]
        index += num_layers
        dynamic_norm_contribs = local[index : index + num_layers]
        index += num_layers
        grad_gates = local[index]

        flat_hidden_pre = grad_hidden_pre.flatten(0, 1).float()
        flat_previous = previous.flatten(0, 1).float()
        grad_hidden_weight = flat_hidden_pre.t() @ flat_previous
        grad_hidden_bias = flat_hidden_pre.sum(0)
        grad_hidden_norm_weight = grad_hidden_norm_contrib.sum(
            tuple(range(grad_hidden_norm_contrib.ndim - 1))
        )

        dynamic_parameter_grads = []
        for layer_input, grad_pre, norm_contrib, (weight, _, _) in zip(
            layer_inputs,
            dynamic_pre_grads,
            dynamic_norm_contribs,
            dynamic,
        ):
            dynamic_parameter_grads.extend(
                (
                    _block_weight_grad(layer_input, grad_pre, weight),
                    grad_pre.float().sum(tuple(range(grad_pre.ndim - 1))),
                    norm_contrib.sum(tuple(range(norm_contrib.ndim - 1))),
                )
            )
        grad_gate_weight = _block_weight_grad(gate_input, grad_gates, gate_weight)
        grad_gate_bias = grad_gates.float().sum(tuple(range(grad_gates.ndim - 1)))

        tensor_grads = (
            grad_projected,
            grad_initial,
            None,
            grad_hidden_weight,
            grad_hidden_bias,
            grad_hidden_norm_weight,
            *dynamic_parameter_grads,
            grad_gate_weight,
            grad_gate_bias,
        )
        return (*tensor_grads, None, None, None, None, None)


class DreamerV3BlockGRUCell(nn.Module):
    """Single-step DreamerV3 block-diagonal GRU cell.

    Args:
        input_size (int): Input feature count.
        hidden_size (int): Recurrent hidden-state width.
        projection_size (int, optional): Input and hidden projection width.
            Defaults to 512.
        num_blocks (int, optional): Number of independent recurrent blocks.
            Defaults to 8.
        num_layers (int, optional): Number of block-linear dynamics layers.
            Defaults to 1.
        activation_class (type[nn.Module] or callable, optional): Parameter-free,
            elementwise, shape-preserving activation. Defaults to :class:`nn.SiLU`.
        norm_eps (float, optional): RMS normalization epsilon. Defaults to ``1e-4``.
        update_bias (float, optional): Fixed update-gate logit offset. Defaults
            to ``-1.0``.
        device (torch.device, optional): Parameter device. Defaults to None.

    Examples:
        >>> import torch
        >>> from torchrl.modules import DreamerV3BlockGRUCell
        >>> cell = DreamerV3BlockGRUCell(6, 8, projection_size=4, num_blocks=2)
        >>> cell(torch.randn(3, 6), torch.zeros(3, 8)).shape
        torch.Size([3, 8])
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        projection_size: int = 512,
        num_blocks: int = 8,
        num_layers: int = 1,
        activation_class: type[nn.Module] | Callable = nn.SiLU,
        norm_eps: float = 1e-4,
        update_bias: float = -1.0,
        device: torch.device | str | int | None = None,
    ):
        super().__init__()
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}.")
        if hidden_size % num_blocks:
            raise ValueError(
                "hidden_size must be divisible by num_blocks, got "
                f"{hidden_size} and {num_blocks}."
            )
        if num_layers < 1:
            raise ValueError(f"num_layers must be positive, got {num_layers}.")
        activation = (
            activation_class()
            if isinstance(activation_class, type)
            else activation_class
        )
        if not callable(activation):
            raise TypeError("activation_class must construct a callable activation.")
        if (
            isinstance(activation, nn.Module)
            and next(activation.parameters(), None) is not None
        ):
            raise ValueError("activation_class must not have learnable parameters.")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.projection_size = projection_size
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.norm_eps = norm_eps
        self.update_bias = update_bias
        self.activation = activation
        self.input_linear = nn.Linear(input_size, projection_size, device=device)
        self.input_norm = _DreamerV3RMSNorm(projection_size, norm_eps, device=device)
        self.hidden_linear = nn.Linear(hidden_size, projection_size, device=device)
        self.hidden_norm = _DreamerV3RMSNorm(projection_size, norm_eps, device=device)

        first_layer_size = hidden_size + 2 * projection_size * num_blocks
        self.dynamic_linears = nn.ModuleList()
        self.dynamic_norms = nn.ModuleList()
        for layer_index in range(num_layers):
            self.dynamic_linears.append(
                _DreamerV3BlockLinear(
                    first_layer_size if layer_index == 0 else hidden_size,
                    hidden_size,
                    num_blocks,
                    device=device,
                )
            )
            self.dynamic_norms.append(
                _DreamerV3RMSNorm(hidden_size, norm_eps, device=device)
            )
        self.gates = _DreamerV3BlockLinear(
            hidden_size, 3 * hidden_size, num_blocks, device=device
        )
        self.apply(_dreamer_v3_init)

    def _run_dynamics(self, value: torch.Tensor) -> torch.Tensor:
        for linear, norm in zip(self.dynamic_linears, self.dynamic_norms):
            value = linear(value)
            value = norm(value)
            value = self.activation(value)
        return value

    def _project_input(self, value: torch.Tensor) -> torch.Tensor:
        value = _dreamer_v3_linear(
            value, self.input_linear.weight, self.input_linear.bias
        )
        value = _dreamer_v3_rms_norm(value, self.input_norm.weight, self.norm_eps)
        return self.activation(value)

    def _step_projected(
        self, projected_input: torch.Tensor, hidden: torch.Tensor
    ) -> torch.Tensor:
        hidden_features = _dreamer_v3_linear(
            hidden, self.hidden_linear.weight, self.hidden_linear.bias
        )
        hidden_features = _dreamer_v3_rms_norm(
            hidden_features, self.hidden_norm.weight, self.norm_eps
        )
        hidden_features = self.activation(hidden_features)
        features = torch.cat((projected_input, hidden_features), -1)
        return _dreamer_v3_block_gru_update(
            features,
            hidden,
            self._run_dynamics,
            self.gates,
            num_blocks=self.num_blocks,
            update_bias=self.update_bias,
        )

    def forward(
        self,
        input: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input.shape[-1] != self.input_size:
            raise ValueError(
                f"Expected input.size(-1) == {self.input_size}, got {input.shape[-1]}."
            )
        if hidden is None:
            hidden = input.new_zeros(*input.shape[:-1], self.hidden_size)
        if hidden.shape != (*input.shape[:-1], self.hidden_size):
            raise ValueError(
                "hidden must match the input batch shape and hidden_size, got "
                f"{hidden.shape}."
            )
        return self._step_projected(self._project_input(input), hidden)


class DreamerV3BlockGRU(nn.Module):
    """Batch-major DreamerV3 block-diagonal GRU sequence module.

    ``is_init`` marks entries whose carry is zeroed before that timestep.
    The ``"reference"`` backend uses ordinary autograd and supports every
    TorchRL-compatible PyTorch version. The opt-in ``"scan"`` backend uses a
    specialized compiled reverse scan.

    Args:
        input_size (int): Input feature count.
        hidden_size (int): Recurrent hidden-state width.
        projection_size (int, optional): Input and hidden projection width.
            Defaults to 512.
        num_blocks (int, optional): Number of independent recurrent blocks.
            Defaults to 8.
        num_layers (int, optional): Number of block-linear dynamics layers.
            Defaults to 1.
        activation_class (type[nn.Module] or callable, optional): Parameter-free,
            elementwise, shape-preserving activation. Defaults to :class:`nn.SiLU`.
        norm_eps (float, optional): RMS normalization epsilon. Defaults to ``1e-4``.
        update_bias (float, optional): Fixed update-gate logit offset. Defaults
            to ``-1.0``.
        recurrent_backend ("reference" or "scan", optional): Sequence backend.
            Defaults to ``"reference"``.
        device (torch.device, optional): Parameter device. Defaults to None.

    Examples:
        >>> import torch
        >>> from torchrl.modules import DreamerV3BlockGRU
        >>> gru = DreamerV3BlockGRU(6, 8, projection_size=4, num_blocks=2)
        >>> output, hidden = gru(torch.randn(3, 5, 6))
        >>> output.shape, hidden.shape
        (torch.Size([3, 5, 8]), torch.Size([3, 8]))
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        projection_size: int = 512,
        num_blocks: int = 8,
        num_layers: int = 1,
        activation_class: type[nn.Module] | Callable = nn.SiLU,
        norm_eps: float = 1e-4,
        update_bias: float = -1.0,
        recurrent_backend: Literal["reference", "scan"] = "reference",
        device: torch.device | str | int | None = None,
    ):
        super().__init__()
        if recurrent_backend not in ("reference", "scan"):
            raise ValueError(
                "recurrent_backend must be 'reference' or 'scan', got "
                f"{recurrent_backend!r}."
            )
        self.cell = DreamerV3BlockGRUCell(
            input_size,
            hidden_size,
            projection_size=projection_size,
            num_blocks=num_blocks,
            num_layers=num_layers,
            activation_class=activation_class,
            norm_eps=norm_eps,
            update_bias=update_bias,
            device=device,
        )
        self.recurrent_backend = recurrent_backend
        if recurrent_backend == "scan":
            _maybe_warm_scan_backward(device)

    def _reference(
        self,
        projected_input: torch.Tensor,
        hidden: torch.Tensor,
        is_init: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = []
        for projected_t, init_t in zip(projected_input.unbind(1), is_init.unbind(1)):
            hidden = torch.where(init_t.unsqueeze(-1), 0, hidden)
            hidden = self.cell._step_projected(projected_t, hidden)
            outputs.append(hidden)
        return torch.stack(outputs, 1), hidden

    def _scan(
        self,
        projected_input: torch.Tensor,
        hidden: torch.Tensor,
        is_init: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        parameters = []
        for linear, norm in zip(self.cell.dynamic_linears, self.cell.dynamic_norms):
            parameters.extend((linear.weight, linear.bias, norm.weight))
        parameters.extend((self.cell.gates.weight, self.cell.gates.bias))
        outputs, final_hidden = _DreamerV3BlockGRUScanFunction.apply(
            projected_input.transpose(0, 1).contiguous(),
            hidden,
            is_init.transpose(0, 1).contiguous(),
            self.cell.hidden_linear.weight,
            self.cell.hidden_linear.bias,
            self.cell.hidden_norm.weight,
            *parameters,
            self.cell.activation,
            self.cell.norm_eps,
            self.cell.update_bias,
            self.cell.num_blocks,
            self.cell.num_layers,
        )
        return outputs.transpose(0, 1), final_hidden

    def forward(
        self,
        input: torch.Tensor,
        hidden: torch.Tensor | None = None,
        is_init: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input.ndim != 3 or input.shape[-1] != self.cell.input_size:
            raise ValueError(
                "input must have shape [batch, time, input_size], got "
                f"{input.shape}."
            )
        batch, time, _ = input.shape
        if time == 0:
            raise ValueError("input must contain at least one timestep.")
        if hidden is None:
            hidden = input.new_zeros(batch, self.cell.hidden_size)
        if hidden.shape != (batch, self.cell.hidden_size):
            raise ValueError(
                f"hidden must have shape {(batch, self.cell.hidden_size)}, got "
                f"{hidden.shape}."
            )
        if is_init is None:
            is_init = torch.zeros(batch, time, dtype=torch.bool, device=input.device)
        elif is_init.ndim == 3 and is_init.shape[-1] == 1:
            is_init = is_init.squeeze(-1)
        if is_init.shape != (batch, time) or is_init.dtype is not torch.bool:
            raise ValueError(
                f"is_init must be boolean with shape {(batch, time)} or "
                f"{(batch, time, 1)}, got {is_init.shape} and {is_init.dtype}."
            )

        projected_input = self.cell._project_input(input.flatten(0, 1)).unflatten(
            0, (batch, time)
        )
        if self.recurrent_backend == "scan":
            return self._scan(projected_input, hidden, is_init)
        return self._reference(projected_input, hidden, is_init)


class DreamerV3MLP(nn.Module):
    """RMS-normalized multilayer perceptron used by DreamerV3 heads.

    Args:
        in_features (int): Input feature count.
        out_features (int): Output feature count.
        depth (int, optional): Number of hidden layers. Defaults to 3.
        num_cells (int, optional): Hidden feature count. Defaults to 1024.
        outscale (float, optional): Multiplicative initialization scale for the
            output layer. Defaults to 1.0.
        norm_eps (float, optional): RMS normalization epsilon. Defaults to
            ``1e-4``.
        device (torch.device, optional): Device on which to create parameters.

    Examples:
        >>> import torch
        >>> from torchrl.modules import DreamerV3MLP
        >>> module = DreamerV3MLP(6, 4, depth=2, num_cells=8)
        >>> module(torch.randn(3, 2), torch.randn(3, 4)).shape
        torch.Size([3, 4])
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        depth: int = 3,
        num_cells: int = 1024,
        outscale: float = 1.0,
        norm_eps: float = 1e-4,
        device=None,
    ):
        super().__init__()
        layers = []
        layer_in = in_features
        for _ in range(depth):
            layers.extend(
                [
                    nn.Linear(layer_in, num_cells, device=device),
                    _DreamerV3RMSNorm(num_cells, norm_eps, device=device),
                    nn.SiLU(),
                ]
            )
            layer_in = num_cells
        output = nn.Linear(layer_in, out_features, device=device)
        layers.append(output)
        self.model = nn.Sequential(*layers)
        self.model.apply(_dreamer_v3_init)
        with torch.no_grad():
            output.weight.mul_(outscale)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        value = inputs[0] if len(inputs) == 1 else torch.cat(inputs, -1)
        return self.model(value)


def _default_bins(
    num_bins: int = _DEFAULT_NUM_BINS,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Build the symmetric raw-value support used by DreamerV3."""
    if num_bins < 2:
        raise ValueError(f"num_bins must be at least 2, got {num_bins}.")
    dtype = dtype or torch.get_default_dtype()
    half_size = num_bins // 2
    if num_bins % 2:
        half = torch.linspace(
            -_DEFAULT_BIN_RANGE,
            0,
            half_size + 1,
            device=device,
            dtype=dtype,
        )
        half = symexp(half)
        return torch.cat((half, -half[:-1].flip(0)))
    step = 2 * _DEFAULT_BIN_RANGE / (num_bins - 1)
    half = -_DEFAULT_BIN_RANGE + step * torch.arange(
        half_size, device=device, dtype=dtype
    )
    half = symexp(half)
    return torch.cat((half, -half.flip(0)))


def _unimix_probs(logits: torch.Tensor, unimix: float) -> torch.Tensor:
    """Return categorical probabilities mixed with a uniform distribution."""
    if not 0 <= unimix < 1:
        raise ValueError(f"unimix must be in [0, 1), got {unimix}.")
    probs = torch.softmax(logits, dim=-1)
    if unimix:
        probs = (1 - unimix) * probs + unimix / logits.shape[-1]
    return probs


def two_hot_encode(x: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Encode raw scalar values on a sorted two-hot support.

    Values between adjacent support points are represented by linear
    interpolation in raw value space. Values outside the support saturate at
    its endpoints.

    Args:
        x (torch.Tensor): Raw scalar targets.
        bins (torch.Tensor): One-dimensional, ascending support.

    Returns:
        A tensor with shape ``(*x.shape, bins.numel())`` on the dtype and
        device of ``x``.

    Examples:
        >>> import torch
        >>> from torchrl.objectives import two_hot_encode
        >>> bins = torch.tensor([-1.0, 0.0, 1.0])
        >>> two_hot_encode(torch.tensor([0.25]), bins)
        tensor([[0.0000, 0.7500, 0.2500]])
    """
    if bins.ndim != 1 or bins.numel() < 2:
        raise ValueError(
            "bins must be a one-dimensional tensor with at least 2 values."
        )
    bins = bins.to(device=x.device, dtype=x.dtype)
    x = x.clamp(bins[0], bins[-1])
    lower = (x.unsqueeze(-1) >= bins).sum(-1) - 1
    lower = lower.clamp(0, bins.numel() - 2)
    upper = lower + 1
    lower_value = bins[lower]
    upper_value = bins[upper]
    upper_weight = (x - lower_value) / (upper_value - lower_value)
    lower_weight = 1 - upper_weight
    target = torch.zeros((*x.shape, bins.numel()), device=x.device, dtype=x.dtype)
    target.scatter_(-1, lower.unsqueeze(-1), lower_weight.unsqueeze(-1))
    target.scatter_(-1, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))
    return target


def two_hot_decode(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Decode logits over a raw-value support to their scalar expectation.

    Args:
        logits (torch.Tensor): Categorical logits whose trailing dimension
            matches the support size.
        bins (torch.Tensor): One-dimensional support in raw value space.

    Returns:
        The softmax-weighted expectation with the trailing category dimension
        removed, preserving the dtype and device of ``logits``.

    Examples:
        >>> import torch
        >>> from torchrl.objectives import two_hot_decode, two_hot_encode
        >>> bins = torch.tensor([-1.0, 0.0, 1.0])
        >>> encoded = two_hot_encode(torch.tensor([0.25]), bins)
        >>> two_hot_decode((encoded + 1e-8).log(), bins)
        tensor([0.2500])
    """
    if bins.ndim != 1 or logits.shape[-1] != bins.numel():
        raise ValueError(
            "The trailing logits dimension must match the one-dimensional support."
        )
    bins = bins.to(device=logits.device, dtype=logits.dtype)
    probs = torch.softmax(logits, dim=-1)
    size = logits.shape[-1]
    if size % 2:
        midpoint = (size - 1) // 2
        center = probs[..., midpoint] * bins[midpoint]
        paired = (
            (probs[..., :midpoint] * bins[:midpoint]).flip(-1)
            + probs[..., midpoint + 1 :] * bins[midpoint + 1 :]
        ).sum(-1)
        return center + paired
    midpoint = size // 2
    return (
        (probs[..., :midpoint] * bins[:midpoint]).flip(-1)
        + probs[..., midpoint:] * bins[midpoint:]
    ).sum(-1)


def two_hot_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    bins: torch.Tensor,
) -> torch.Tensor:
    """Return two-hot cross entropy for raw scalar targets.

    Args:
        logits (torch.Tensor): Categorical logits with bins in the trailing
            dimension.
        target (torch.Tensor): Raw scalar targets, optionally with a trailing
            singleton dimension.
        bins (torch.Tensor): One-dimensional support in raw value space.

    Returns:
        The unreduced cross entropy with the trailing category dimension
        removed.

    Examples:
        >>> import torch
        >>> from torchrl.objectives import two_hot_cross_entropy
        >>> logits = torch.zeros(2, 3)
        >>> target = torch.tensor([-0.5, 0.5])
        >>> two_hot_cross_entropy(logits, target, torch.tensor([-1.0, 0.0, 1.0]))
        tensor([1.0986, 1.0986])
    """
    if target.shape == (*logits.shape[:-1], 1):
        target = target.squeeze(-1)
    if target.shape != logits.shape[:-1]:
        raise ValueError(
            f"target shape must be {logits.shape[:-1]} or "
            f"{(*logits.shape[:-1], 1)}, got {target.shape}."
        )
    encoded = two_hot_encode(target, bins)
    return -(encoded * torch.log_softmax(logits, dim=-1)).sum(-1)


class SymExpTwoHot(nn.Module):
    """DreamerV3 categorical scalar representation.

    The support contains ``num_bins`` raw scalar values obtained by applying
    ``symexp`` to an evenly spaced grid from -20 to 20. Targets are interpolated
    between adjacent raw support values, while predictions are decoded as the
    softmax-weighted raw-value expectation.

    Args:
        num_bins (int, optional): Number of categorical support values.
            Defaults to 255.

    Examples:
        >>> import torch
        >>> from torchrl.modules import SymExpTwoHot
        >>> two_hot = SymExpTwoHot(num_bins=5)
        >>> target = torch.tensor([-10.0, 0.0, 10.0])
        >>> encoded = two_hot.encode(target)
        >>> decoded = two_hot.decode(encoded.log())
        >>> torch.allclose(decoded, target, atol=1e-3)
        True
    """

    def __init__(self, num_bins: int = _DEFAULT_NUM_BINS):
        super().__init__()
        self.num_bins = num_bins
        self.register_buffer("bins", _default_bins(num_bins))

    def encode(self, target: torch.Tensor) -> torch.Tensor:
        """Encode raw scalar targets as two-hot categorical targets."""
        return two_hot_encode(target, self.bins)

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        """Decode categorical logits to raw scalar values."""
        return two_hot_decode(logits, self.bins)

    def loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute two-hot cross entropy against raw scalar targets."""
        return two_hot_cross_entropy(logits, target, self.bins)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Decode logits and retain a trailing scalar event dimension."""
        return self.decode(logits).unsqueeze(-1)


class RSSMPriorV3(nn.Module):
    """DreamerV3 prior network with discrete categorical latent state.

    See :doc:`DreamerV3 in a nutshell </reference/dreamer_v3>` for the prior's
    role in observation-conditioned filtering and latent imagination.

    Implements the sequence model and dynamics predictor from DreamerV3.
    The GRU updates the deterministic hidden state:

    .. code-block:: text

        h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])

    Then the prior predicts a distribution over the stochastic latent:

    .. code-block:: text

        z_hat_t ~ Cat(MLP(h_t))

    Reference: https://arxiv.org/abs/2301.04104

    Args:
        action_spec (TensorSpec, optional): Action spec. Used only to read
            ``action_spec.shape``; mutually exclusive with ``action_shape``.
        action_shape (torch.Size or tuple of int, optional): Action tensor
            shape. Mutually exclusive with ``action_spec``.
        hidden_dim (int, optional): Hidden dimension of the linear projector.
            Defaults to 512.
        rnn_hidden_dim (int, optional): GRU hidden state dimension (belief size).
            Defaults to 512.
        num_categoricals (int, optional): Number of categorical variables in the
            discrete latent. Defaults to 32.
        num_classes (int, optional): Number of classes per categorical variable.
            Defaults to 32.
        action_dim (int, optional): Action dimension. If provided (along with
            ``num_categoricals * num_classes``), uses explicit ``nn.Linear``
            instead of ``nn.LazyLinear``. Defaults to None.
        recurrent_model ("gru" or "block_gru", optional): Recurrent core.
            ``"gru"`` preserves the historical TorchRL implementation while
            ``"block_gru"`` selects the grouped DreamerV3 core. Defaults to
            ``"gru"``.
        num_blocks (int, optional): Number of groups in the block GRU.
            Defaults to 8.
        num_layers (int, optional): Number of block-linear dynamics layers.
            Defaults to 1.
        prior_num_layers (int, optional): Number of prior predictor layers in
            block-GRU mode. Defaults to 2.
        norm_eps (float, optional): RMS normalization epsilon. Defaults to
            ``1e-4``.
        unimix (float, optional): Fraction of uniform probability mixed into
            categorical samples. Defaults to ``0.0`` for compatibility.
        device (torch.device, optional): Device. Defaults to None.

    Examples:
        >>> import torch
        >>> from torchrl.modules.models.model_based import RSSMPriorV3
        >>> prior = RSSMPriorV3(
        ...     action_shape=torch.Size([2]),
        ...     hidden_dim=16,
        ...     rnn_hidden_dim=8,
        ...     num_categoricals=4,
        ...     num_classes=4,
        ...     action_dim=2,
        ... )
        >>> state = torch.zeros(3, 16)
        >>> belief = torch.zeros(3, 8)
        >>> action = torch.randn(3, 2)
        >>> logits, next_state, next_belief = prior(state, belief, action)
        >>> logits.shape, next_state.shape, next_belief.shape
        (torch.Size([3, 4, 4]), torch.Size([3, 16]), torch.Size([3, 8]))
    """

    def __init__(
        self,
        action_spec=None,
        hidden_dim: int = 512,
        rnn_hidden_dim: int = 512,
        num_categoricals: int = 32,
        num_classes: int = 32,
        action_dim: int | None = None,
        device=None,
        *,
        action_shape: torch.Size | tuple[int, ...] | None = None,
        recurrent_model: Literal["gru", "block_gru"] = "gru",
        num_blocks: int = 8,
        num_layers: int = 1,
        prior_num_layers: int = 2,
        norm_eps: float = 1e-4,
        unimix: float = 0.0,
    ):
        super().__init__()
        if action_spec is not None and action_shape is not None:
            raise ValueError(
                "Pass only one of `action_spec` or `action_shape`, not both."
            )
        if action_spec is not None:
            self.action_shape = torch.Size(action_spec.shape)
        elif action_shape is not None:
            self.action_shape = torch.Size(action_shape)
        else:
            self.action_shape = None

        self.num_categoricals = num_categoricals
        self.num_classes = num_classes
        self.rnn_hidden_dim = rnn_hidden_dim
        if not 0 <= unimix < 1:
            raise ValueError(f"unimix must be in [0, 1), got {unimix}.")
        self.unimix = unimix
        state_dim = num_categoricals * num_classes

        if recurrent_model not in ("gru", "block_gru"):
            raise ValueError(
                "recurrent_model must be 'gru' or 'block_gru', got "
                f"{recurrent_model!r}."
            )
        if recurrent_model == "block_gru" and action_dim is None:
            raise ValueError("block_gru requires an explicit action_dim.")
        self.recurrent_model = recurrent_model

        if recurrent_model == "block_gru":
            self.rnn = _DreamerV3BlockGRU(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                belief_dim=rnn_hidden_dim,
                num_blocks=num_blocks,
                num_layers=num_layers,
                norm_eps=norm_eps,
                device=device,
            )
            prior_layers = []
            prior_in = rnn_hidden_dim
            for _ in range(prior_num_layers):
                prior_layers.extend(
                    [
                        nn.Linear(prior_in, hidden_dim, device=device),
                        _DreamerV3RMSNorm(hidden_dim, norm_eps, device=device),
                        nn.SiLU(),
                    ]
                )
                prior_in = hidden_dim
            prior_layers.append(
                nn.Linear(
                    prior_in,
                    num_categoricals * num_classes,
                    device=device,
                )
            )
            self.rnn_to_prior_projector = nn.Sequential(*prior_layers)
            self.rnn_to_prior_projector.apply(_dreamer_v3_init)
            self.action_state_projector = None
        else:
            self.rnn = GRUCell(hidden_dim, rnn_hidden_dim, device=device)
            if action_dim is not None:
                projector_in = state_dim + action_dim
                first_linear = nn.Linear(projector_in, hidden_dim, device=device)
            else:
                first_linear = nn.LazyLinear(hidden_dim, device=device)
            self.action_state_projector = nn.Sequential(first_linear, nn.SiLU())
            self.rnn_to_prior_projector = nn.Sequential(
                nn.Linear(rnn_hidden_dim, hidden_dim, device=device),
                nn.SiLU(),
                nn.Linear(hidden_dim, num_categoricals * num_classes, device=device),
            )

    def forward(
        self,
        state: torch.Tensor,
        belief: torch.Tensor,
        action: torch.Tensor,
        *,
        _uniform: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute prior distribution and update GRU belief.

        Args:
            state: Previous stochastic state, shape ``[..., num_categoricals * num_classes]``.
            belief: Previous GRU hidden state, shape ``[..., rnn_hidden_dim]``.
            action: Current action, shape ``[..., action_dim]``.
            _uniform: Optional pre-sampled uniforms used by the scan backend.

        Returns:
            prior_logits (torch.Tensor): Raw logits, shape
                ``[..., num_categoricals, num_classes]``.
            state (torch.Tensor): Sampled state (straight-through), shape
                ``[..., num_categoricals * num_classes]``.
            belief (torch.Tensor): Updated GRU hidden state, shape
                ``[..., rnn_hidden_dim]``.
        """
        prior_logits, belief = self._belief_and_logits(state, belief, action)
        state = _straight_through_categorical(
            prior_logits, self.unimix, uniform=_uniform
        )
        state = state.view(*state.shape[:-2], self.num_categoricals * self.num_classes)

        return prior_logits, state, belief

    def _belief_and_logits(
        self,
        state: torch.Tensor,
        belief: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the prior logits and the new belief, without sampling."""
        belief = self._update_belief(state, belief, action)
        prior_logits_flat = self.rnn_to_prior_projector(belief)
        prior_logits = prior_logits_flat.view(
            *prior_logits_flat.shape[:-1], self.num_categoricals, self.num_classes
        )
        return prior_logits, belief

    def _update_belief(
        self,
        state: torch.Tensor,
        belief: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Advance the deterministic state and skip the prior head.

        The acting path conditions on the observation, never on a prior sample.
        """
        if self.recurrent_model == "block_gru":
            belief = self.rnn(state, belief, action)
        else:
            projector_input = torch.cat([state, action], dim=-1)
            action_state = self.action_state_projector(projector_input)

            # Run GRU in fp32 to avoid cuBLAS dispatch issues under autocast
            dtype = action_state.dtype
            device_type = action_state.device.type
            with torch.amp.autocast(device_type=device_type, enabled=False):
                belief = self.rnn(
                    action_state.float(),
                    belief.float() if belief is not None else None,
                )
            belief = belief.to(dtype)
        return belief


class RSSMPosteriorV3(nn.Module):
    """DreamerV3 posterior (representation model) with discrete categorical latent.

    See :doc:`DreamerV3 in a nutshell </reference/dreamer_v3>` for the
    relationship between the posterior, prior, stochastic state, and belief.

    Given the deterministic hidden state ``h_t`` and an observation embedding
    ``e_t``, produces the posterior distribution over the stochastic latent:

    .. code-block:: text

        z_t ~ Cat(MLP([h_t, e_t]))

    Reference: https://arxiv.org/abs/2301.04104

    Args:
        hidden_dim (int, optional): Hidden dimension of the projector MLP.
            Defaults to 512.
        num_categoricals (int, optional): Number of categorical variables.
            Defaults to 32.
        num_classes (int, optional): Number of classes per categorical variable.
            Defaults to 32.
        rnn_hidden_dim (int, optional): Belief dimension. If provided along with
            ``obs_embed_dim``, uses explicit ``nn.Linear``. Defaults to None.
        obs_embed_dim (int, optional): Observation embedding dimension. If provided
            along with ``rnn_hidden_dim``, uses explicit ``nn.Linear``. Defaults to None.
        use_rms_norm (bool, optional): Build the observation predictor from
            RMS-normalized DreamerV3 layers. Defaults to ``False`` for checkpoint
            compatibility.
        num_layers (int, optional): Number of observation predictor layers when
            ``use_rms_norm=True``. Defaults to 1.
        norm_eps (float, optional): RMS normalization epsilon. Defaults to
            ``1e-4``.
        unimix (float, optional): Fraction of uniform probability mixed into
            categorical samples. Defaults to ``0.0`` for compatibility.
        device (torch.device, optional): Device. Defaults to None.

    Examples:
        >>> import torch
        >>> from torchrl.modules.models.model_based import RSSMPosteriorV3
        >>> posterior = RSSMPosteriorV3(
        ...     hidden_dim=16,
        ...     num_categoricals=4,
        ...     num_classes=4,
        ...     rnn_hidden_dim=8,
        ...     obs_embed_dim=12,
        ... )
        >>> belief = torch.randn(3, 8)
        >>> obs_embed = torch.randn(3, 12)
        >>> logits, state = posterior(belief, obs_embed)
        >>> logits.shape, state.shape
        (torch.Size([3, 4, 4]), torch.Size([3, 16]))
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_categoricals: int = 32,
        num_classes: int = 32,
        rnn_hidden_dim: int | None = None,
        obs_embed_dim: int | None = None,
        device=None,
        *,
        use_rms_norm: bool = False,
        num_layers: int = 1,
        norm_eps: float = 1e-4,
        unimix: float = 0.0,
    ):
        super().__init__()
        self.num_categoricals = num_categoricals
        self.num_classes = num_classes
        if not 0 <= unimix < 1:
            raise ValueError(f"unimix must be in [0, 1), got {unimix}.")
        self.unimix = unimix

        if use_rms_norm and (rnn_hidden_dim is None or obs_embed_dim is None):
            raise ValueError(
                "use_rms_norm=True requires explicit rnn_hidden_dim and "
                "obs_embed_dim."
            )

        if rnn_hidden_dim is not None and obs_embed_dim is not None:
            projector_in = rnn_hidden_dim + obs_embed_dim
            first_linear = nn.Linear(projector_in, hidden_dim, device=device)
        else:
            first_linear = nn.LazyLinear(hidden_dim, device=device)

        if use_rms_norm:
            layers = []
            layer_in = projector_in
            for layer_index in range(num_layers):
                linear = (
                    first_linear
                    if layer_index == 0
                    else nn.Linear(layer_in, hidden_dim, device=device)
                )
                layers.extend(
                    [
                        linear,
                        _DreamerV3RMSNorm(hidden_dim, norm_eps, device=device),
                        nn.SiLU(),
                    ]
                )
                layer_in = hidden_dim
            layers.append(
                nn.Linear(
                    layer_in,
                    num_categoricals * num_classes,
                    device=device,
                )
            )
            self.obs_rnn_to_post_projector = nn.Sequential(*layers)
            self.obs_rnn_to_post_projector.apply(_dreamer_v3_init)
        else:
            self.obs_rnn_to_post_projector = nn.Sequential(
                first_linear,
                nn.SiLU(),
                nn.Linear(hidden_dim, num_categoricals * num_classes, device=device),
            )

    def forward(
        self,
        belief: torch.Tensor,
        obs_embedding: torch.Tensor,
        *,
        _uniform: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute posterior distribution given belief and observation embedding.

        Args:
            belief: Deterministic GRU hidden state from prior, shape
                ``[..., rnn_hidden_dim]``.
            obs_embedding: Encoded observation, shape ``[..., obs_embed_dim]``.
            _uniform: Optional pre-sampled uniforms used by the scan backend.

        Returns:
            posterior_logits (torch.Tensor): Raw logits, shape
                ``[..., num_categoricals, num_classes]``.
            state (torch.Tensor): Sampled state (straight-through), shape
                ``[..., num_categoricals * num_classes]``.
        """
        posterior_logits = self._logits(belief, obs_embedding)
        state = _straight_through_categorical(
            posterior_logits, self.unimix, uniform=_uniform
        )
        state = state.view(*state.shape[:-2], self.num_categoricals * self.num_classes)
        return posterior_logits, state

    def _logits(
        self, belief: torch.Tensor, obs_embedding: torch.Tensor
    ) -> torch.Tensor:
        post_logits_flat = self.obs_rnn_to_post_projector(
            torch.cat([belief, obs_embedding], dim=-1)
        )
        return post_logits_flat.view(
            *post_logits_flat.shape[:-1], self.num_categoricals, self.num_classes
        )


class RSSMRolloutV3(TensorDictModuleBase):
    """Roll out the DreamerV3 RSSM over a sequence.

    See :doc:`DreamerV3 in a nutshell </reference/dreamer_v3>` for the RSSM
    data flow and terminology used by this rollout.

    Given encoded observations and actions for ``T`` time steps, this module
    runs the prior (GRU + categorical) then the posterior (categorical) at each
    step and returns a stacked TensorDict of all intermediate states.

    The previous posterior state ``z_t`` is used as the prior input for step
    ``t+1``, matching the recurrent structure of DreamerV3.

    The module picks one of two paths at construction: tensors when the
    modules use the standard DreamerV3 key wiring, TensorDicts otherwise. Both
    give identical results, and the tensor path shares storage for the entries
    it does not overwrite. See :meth:`compile_rollout`.

    Reference: https://arxiv.org/abs/2301.04104

    Args:
        rssm_prior (TensorDictModule): Prior module wrapping :class:`RSSMPriorV3`.
        rssm_posterior (TensorDictModule): Posterior module wrapping
            :class:`RSSMPosteriorV3`.
        reset_key (NestedKey or None, optional): Boolean key marking the first
            transition of an episode. The rollout zeroes the state, belief and
            action there. Defaults to ``"is_init"``.
        action_key (NestedKey or None, optional): Action key, zeroed on a reset
            step. Defaults to ``None``: the module then takes the
            ``rssm_prior`` input key that is not ``"state"`` or ``"belief"``.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.modules.models.model_based import (
        ...     RSSMPosteriorV3, RSSMPriorV3, RSSMRolloutV3,
        ... )
        >>> prior = TensorDictModule(
        ...     RSSMPriorV3(action_shape=torch.Size([2]), hidden_dim=8,
        ...                 rnn_hidden_dim=8, num_categoricals=4, num_classes=4,
        ...                 action_dim=2),
        ...     in_keys=["state", "belief", "action"],
        ...     out_keys=[("next", "prior_logits"), ("next", "state"), ("next", "belief")],
        ... )
        >>> posterior = TensorDictModule(
        ...     RSSMPosteriorV3(hidden_dim=8, num_categoricals=4, num_classes=4,
        ...                     rnn_hidden_dim=8, obs_embed_dim=6),
        ...     in_keys=[("next", "belief"), ("next", "encoded_latents")],
        ...     out_keys=[("next", "posterior_logits"), ("next", "state")],
        ... )
        >>> rollout = RSSMRolloutV3(prior, posterior)
        >>> td = TensorDict({
        ...     "state": torch.zeros(2, 4, 16),
        ...     "belief": torch.zeros(2, 4, 8),
        ...     "action": torch.randn(2, 4, 2),
        ...     "next": {"encoded_latents": torch.randn(2, 4, 6)},
        ... }, [2, 4])
        >>> out = rollout(td)
        >>> out.shape
        torch.Size([2, 4])
    """

    def __init__(
        self,
        rssm_prior: TensorDictModule,
        rssm_posterior: TensorDictModule,
        reset_key: NestedKey | None = "is_init",
        action_key: NestedKey | None = None,
    ):
        super().__init__()
        _module = TensorDictSequential(rssm_prior, rssm_posterior)
        self.in_keys = _module.in_keys
        self.out_keys = _module.out_keys
        self.rssm_prior = rssm_prior
        self.rssm_posterior = rssm_posterior
        self.reset_key = unravel_key(reset_key) if reset_key is not None else None
        if action_key is not None:
            self.action_key = unravel_key(action_key)
        else:
            candidates = [
                key
                for key in map(unravel_key, rssm_prior.in_keys)
                if key not in ("state", "belief")
            ]
            if len(candidates) > 1:
                raise ValueError(
                    "Could not infer the action key from the prior in_keys "
                    f"{list(rssm_prior.in_keys)}: {candidates} are all "
                    "candidates. Pass action_key explicitly."
                )
            self.action_key = candidates[0] if candidates else None

        self._fast_path = self._check_fast_path()
        self._step_fn = None
        self._scan_fn = None

    def _check_fast_path(self) -> bool:
        """Return ``True`` for the standard DreamerV3 key wiring."""

        def keys(module_keys):
            return [unravel_key(key) for key in module_keys]

        # The tensor path calls the modules by position: only action is free.
        return (
            type(getattr(self.rssm_prior, "module", None)) is RSSMPriorV3
            and type(getattr(self.rssm_posterior, "module", None)) is RSSMPosteriorV3
            and keys(self.rssm_prior.in_keys) == ["state", "belief", self.action_key]
            and keys(self.rssm_prior.out_keys)
            == [
                ("next", "prior_logits"),
                ("next", "state"),
                ("next", "belief"),
            ]
            and keys(self.rssm_posterior.in_keys)
            == [("next", "belief"), ("next", "encoded_latents")]
            and keys(self.rssm_posterior.out_keys)
            == [("next", "posterior_logits"), ("next", "state")]
        )

    def forward(self, tensordict):
        """Roll out the RSSM for one episode chunk.

        Args:
            tensordict (TensorDictBase): Input with shape ``[*batch, T]`` containing
                actions, encoded observations, and initial state/belief.

        Returns:
            TensorDictBase: Stacked outputs with shape ``[*batch, T]``.
        """
        if self._fast_path:
            return self._forward_fast(tensordict)

        tensordict_out = []
        *batch, time_steps = tensordict.shape

        update_values = tensordict.exclude(*self.out_keys).unbind(-1)
        _tensordict = update_values[0]

        # Cache the keys we want to keep; they're constant across timesteps.
        output_keys = list(
            update_values[0].keys(include_nested=True, leaves_only=True)
        ) + list(self.out_keys)

        for t in range(time_steps):
            reset = (
                _tensordict.get(self.reset_key, None)
                if self.reset_key is not None
                else None
            )
            if reset is not None:
                state = _tensordict.get("state")
                belief = _tensordict.get("belief")
                while reset.ndim < state.ndim:
                    reset = reset.unsqueeze(-1)
                _tensordict.set("state", torch.where(reset, 0, state))
                _tensordict.set("belief", torch.where(reset, 0, belief))
                # A reset step must not use the previous action either.
                action = (
                    _tensordict.get(self.action_key, None)
                    if self.action_key is not None
                    else None
                )
                if action is not None:
                    action_reset = reset
                    while action_reset.ndim > action.ndim:
                        action_reset = action_reset.squeeze(-1)
                    while action_reset.ndim < action.ndim:
                        action_reset = action_reset.unsqueeze(-1)
                    _tensordict.set(
                        self.action_key, torch.where(action_reset, 0, action)
                    )
            self.rssm_prior(_tensordict)
            self.rssm_posterior(_tensordict)

            tensordict_out.append(_tensordict.select(*output_keys, strict=False))
            if t < time_steps - 1:
                next_state = _tensordict.get(("next", "state"))
                next_belief = _tensordict.get(("next", "belief"))
                _tensordict = update_values[t + 1]
                _tensordict.set("state", next_state)
                _tensordict.set("belief", next_belief)

        return torch.stack(tensordict_out, tensordict.ndim - 1)

    def _forward_fast(self, tensordict):
        """Run the recurrence on tensors, writing the TensorDict once."""
        action = tensordict.get(self.action_key)
        embedding = tensordict.get(("next", "encoded_latents"))
        state = tensordict.get("state")[..., 0, :].contiguous()
        belief = tensordict.get("belief")[..., 0, :].contiguous()
        reset = (
            tensordict.get(self.reset_key, None) if self.reset_key is not None else None
        )
        if reset is None:
            reset = torch.zeros_like(action[..., :1], dtype=torch.bool)
        while reset.ndim > action.ndim and reset.shape[-1] == 1:
            reset = reset.squeeze(-1)
        while reset.ndim < action.ndim:
            reset = reset.unsqueeze(-1)

        scan = self._scan_fn or self._loop
        (
            input_states,
            input_beliefs,
            masked_actions,
            prior_logits,
            posterior_logits,
            next_states,
            next_beliefs,
        ) = scan(state, belief, action, embedding, reset)

        # Write back the masked inputs, as the TensorDict path does.
        output = tensordict.exclude(*self.out_keys)
        output.set("state", input_states)
        output.set("belief", input_beliefs)
        output.set(self.action_key, masked_actions)
        output.set(("next", "prior_logits"), prior_logits)
        output.set(("next", "posterior_logits"), posterior_logits)
        output.set(("next", "state"), next_states)
        output.set(("next", "belief"), next_beliefs)
        return output

    def _step(self, state, belief, action_t, embedding_t, reset_t):
        """Run one deterministic step of the recurrence.

        The two categorical draws stay outside, in :meth:`_scan`, so that
        :func:`torch.compile` leaves the random stream unchanged.
        """
        prior_net = self.rssm_prior.module
        posterior_net = self.rssm_posterior.module
        state = torch.where(reset_t, 0, state)
        belief = torch.where(reset_t, 0, belief)
        action_t = torch.where(reset_t, 0, action_t)
        prior_logits_t, next_belief = prior_net._belief_and_logits(
            state, belief, action_t
        )
        posterior_logits_t = posterior_net._logits(next_belief, embedding_t)
        return state, belief, action_t, prior_logits_t, next_belief, posterior_logits_t

    def _loop(self, state, belief, action, embedding, reset):
        """Run the recurrence with an explicit Python loop."""
        prior_net = self.rssm_prior.module
        posterior_net = self.rssm_posterior.module
        step = self._step_fn or self._step
        uniforms = torch.rand(
            action.shape[-2],
            2,
            *action.shape[:-2],
            prior_net.num_categoricals,
            device=action.device,
        )
        input_states = []
        input_beliefs = []
        masked_actions = []
        prior_logits = []
        posterior_logits = []
        next_states = []
        next_beliefs = []

        for time_index in range(action.shape[-2]):
            (
                masked_state,
                masked_belief,
                action_t,
                prior_logits_t,
                belief,
                posterior_logits_t,
            ) = step(
                state,
                belief,
                action[..., time_index, :],
                embedding[..., time_index, :],
                reset[..., time_index, :],
            )
            input_states.append(masked_state)
            input_beliefs.append(masked_belief)
            masked_actions.append(action_t)

            # Discarded draw: it keeps the random stream equal to the TD path.
            _straight_through_categorical(
                prior_logits_t, prior_net.unimix, uniforms[time_index, 0]
            )
            state = _straight_through_categorical(
                posterior_logits_t,
                posterior_net.unimix,
                uniforms[time_index, 1],
            )
            state = state.view(
                *state.shape[:-2],
                posterior_net.num_categoricals * posterior_net.num_classes,
            )
            prior_logits.append(prior_logits_t)
            posterior_logits.append(posterior_logits_t)
            next_states.append(state)
            next_beliefs.append(belief)

        return (
            torch.stack(input_states, -2),
            torch.stack(input_beliefs, -2),
            torch.stack(masked_actions, -2),
            torch.stack(prior_logits, -3),
            torch.stack(posterior_logits, -3),
            torch.stack(next_states, -2),
            torch.stack(next_beliefs, -2),
        )

    def _scan(self, state, belief, action, embedding, reset, *, unroll: int = 1):
        """Run the recurrence with the higher-order :func:`torch.scan`."""
        if not isinstance(unroll, int) or isinstance(unroll, bool) or unroll < 1:
            raise ValueError(f"unroll must be a positive integer, got {unroll!r}.")
        prior_net = self.rssm_prior.module
        posterior_net = self.rssm_posterior.module
        length = action.shape[-2]
        uniforms = torch.rand(
            length,
            2,
            *action.shape[:-2],
            prior_net.num_categoricals,
            device=action.device,
        )

        def step(carry, xs):
            state, belief = carry
            (
                action_t,
                embedding_t,
                reset_t,
                prior_uniform,
                posterior_uniform,
            ) = xs
            state = torch.where(reset_t, 0, state)
            belief = torch.where(reset_t, 0, belief)
            action_t = torch.where(reset_t, 0, action_t)
            masked_state = state
            masked_belief = belief
            prior_logits, _, belief = prior_net(
                state, belief, action_t, _uniform=prior_uniform
            )
            posterior_logits, state = posterior_net(
                belief, embedding_t, _uniform=posterior_uniform
            )
            output = (
                masked_state.clone(),
                masked_belief.clone(),
                action_t,
                prior_logits,
                posterior_logits,
                state.clone(),
                belief.clone(),
            )
            return (state.clone(), belief.clone()), output

        scan_inputs = (
            action.movedim(-2, 0),
            embedding.movedim(-2, 0),
            reset.movedim(-2, 0),
            uniforms[:, 0],
            uniforms[:, 1],
        )
        if unroll > 1:
            padding = (-length) % unroll
            if padding:
                scan_inputs = tuple(
                    torch.cat(
                        (
                            value,
                            value.new_zeros((padding, *value.shape[1:])),
                        ),
                        dim=0,
                    )
                    for value in scan_inputs
                )
            padded_length = length + padding
            scan_inputs = tuple(
                value.reshape(padded_length // unroll, unroll, *value.shape[1:])
                for value in scan_inputs
            )

            def combine(carry, xs):
                outputs = tuple([] for _ in range(7))
                for index in range(unroll):
                    carry, output = step(carry, tuple(value[index] for value in xs))
                    for output_list, value in zip(outputs, output):
                        output_list.append(value)
                return carry, tuple(
                    torch.stack(output_list, 0) for output_list in outputs
                )

        else:
            combine = step

        # Keep the module weights as closure inputs, as the recurrent modules
        # do. The higher-order operator lifts them once instead of scanning
        # time-expanded parameter views at every step.
        _, output = _higher_order_scan(
            combine,
            (state, belief),
            scan_inputs,
            dim=0,
        )
        if unroll > 1:
            output = tuple(value.flatten(0, 1)[:length] for value in output)
        (
            input_states,
            input_beliefs,
            masked_actions,
            prior_logits,
            posterior_logits,
            next_states,
            next_beliefs,
        ) = output
        return (
            input_states.movedim(0, -2),
            input_beliefs.movedim(0, -2),
            masked_actions.movedim(0, -2),
            prior_logits.movedim(0, -3),
            posterior_logits.movedim(0, -3),
            next_states.movedim(0, -2),
            next_beliefs.movedim(0, -2),
        )

    def compile_rollout(
        self,
        scope: Literal["step", "scan"] = "step",
        *,
        unroll: int = 1,
        **compile_kwargs,
    ) -> None:
        """Compile the recurrence with :func:`torch.compile`.

        ``"step"`` compiles one deterministic step of the default explicit
        loop. ``"scan"`` selects and compiles the higher-order scan backend.
        Random samples are supplied as higher-order scan inputs. Eager and
        compiled executions are not expected to consume identical RNG streams.

        Both scopes need the tensor path.

        Args:
            scope ("step" or "scan", optional): Part of the recurrence to
                compile. Defaults to ``"step"``.
            unroll (int, optional): Number of scan steps to trace in each
                higher-order scan iteration. Larger values can improve runtime
                at the cost of compilation time and graph size. Only applies
                to ``scope="scan"``. Defaults to ``1``.
            **compile_kwargs: Keyword arguments for :func:`torch.compile`.
                ``dynamic`` defaults to ``False``.
        """
        if not self._fast_path:
            raise RuntimeError(
                "compile_rollout() requires the tensor path, which needs the "
                "standard DreamerV3 module wiring."
            )
        if scope not in ("step", "scan"):
            raise ValueError(f"scope must be 'step' or 'scan', got {scope!r}.")
        if not isinstance(unroll, int) or isinstance(unroll, bool) or unroll < 1:
            raise ValueError(f"unroll must be a positive integer, got {unroll!r}.")
        if scope != "scan" and unroll != 1:
            raise ValueError("unroll only applies when scope='scan'.")
        compile_kwargs.setdefault("dynamic", False)
        self._step_fn = self._scan_fn = None
        if scope == "step":
            self._step_fn = torch.compile(self._step, **compile_kwargs)
        else:
            devices = {value.device for value in self.parameters()}
            devices.update(value.device for value in self.buffers())
            for device in devices:
                _maybe_warm_scan_backward(device)
            self._scan_fn = torch.compile(
                ft.partial(self._scan, unroll=unroll), **compile_kwargs
            )

    def __getstate__(self) -> dict:
        # Pickle cannot store a compiled callable: the copy starts eager.
        state = super().__getstate__()
        state["_step_fn"] = state["_scan_fn"] = None
        return state


def _straight_through_categorical(
    logits: torch.Tensor,
    unimix: float = 0.0,
    uniform: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample from categorical with straight-through gradient estimator.

    Forward: hard one-hot sample.
    Backward: gradients flow through the soft probabilities.

    Args:
        logits: ``[..., num_categoricals, num_classes]``
        unimix: Weight of the uniform-probability mixture.
        uniform: Optional ``[..., num_categoricals]`` uniform samples. Passing
            them explicitly keeps higher-order scans pure and reproducible.

    Returns:
        one_hot tensor with same shape, gradients through softmax.
    """
    probs = _unimix_probs(logits, unimix)
    if uniform is None:
        uniform = torch.rand(probs.shape[:-1], device=probs.device)
    indices = (uniform.unsqueeze(-1) > probs.cumsum(-1)).sum(-1)
    indices = indices.clamp_max(probs.shape[-1] - 1)
    one_hot = torch.zeros_like(probs)
    one_hot.scatter_(-1, indices.unsqueeze(-1), 1.0)
    # Straight-through: forward = one_hot, backward gradient = grad(probs).
    return probs + (one_hot - probs).detach()


class DreamerActor(nn.Module):
    """Dreamer actor network.

    This network is used to predict the action distribution given the
    the stochastic state and the deterministic belief at the current
    time step.
    It outputs the mean and the scale of the action distribution.

    Reference: https://arxiv.org/abs/1912.01603

    Args:
        out_features (int): Number of output features.
        depth (int, optional): Number of hidden layers.
            Defaults to 4.
        num_cells (int, optional): Number of hidden units per layer.
            Defaults to 200.
        activation_class (nn.Module, optional): Activation class.
            Defaults to nn.ELU.
        std_bias (:obj:`float`, optional): Bias of the softplus transform.
            Defaults to 5.0.
        std_min_val (:obj:`float`, optional): Minimum value of the standard deviation.
            Defaults to 1e-4.
        device (torch.device, optional): Device to create the module on.
            Defaults to None (uses default device).
    """

    def __init__(
        self,
        out_features,
        depth=4,
        num_cells=200,
        activation_class=nn.ELU,
        std_bias=5.0,
        std_min_val=1e-4,
        device=None,
    ):
        super().__init__()
        self.backbone = MLP(
            out_features=2 * out_features,
            depth=depth,
            num_cells=num_cells,
            activation_class=activation_class,
            device=device,
        )
        self.backbone.append(
            NormalParamExtractor(
                scale_mapping=f"biased_softplus_{std_bias}_{std_min_val}",
                # scale_mapping="relu",
            ),
        )

    def forward(self, state, belief):
        loc, scale = self.backbone(state, belief)
        return loc, scale


class ObsEncoder(nn.Module):
    """Observation encoder network.

    Takes a pixel observation and encodes it into a latent space.

    Reference: https://arxiv.org/abs/1803.10122

    Args:
        channels (int, optional): Number of hidden units in the first layer.
            Defaults to 32.
        num_layers (int, optional): Depth of the network. Defaults to 4.
        in_channels (int, optional): Number of input channels. If None, uses LazyConv2d.
            Defaults to None for backward compatibility.
        device (torch.device, optional): Device to create the module on.
            Defaults to None (uses default device).
    """

    def __init__(
        self, channels=32, num_layers=4, in_channels=None, depth=None, device=None
    ):
        if depth is not None:
            warnings.warn(
                f"The depth argument in {type(self)} will soon be deprecated and "
                f"used for the depth of the network instead. Please use channels "
                f"for the layer size and num_layers for the depth until depth "
                f"replaces num_layers."
            )
            channels = depth
        if num_layers < 1:
            raise RuntimeError("num_layers cannot be smaller than 1.")
        super().__init__()
        # Use explicit Conv2d if in_channels provided, else LazyConv2d for backward compat
        if in_channels is not None:
            first_conv = nn.Conv2d(in_channels, channels, 4, stride=2, device=device)
        else:
            first_conv = nn.LazyConv2d(channels, 4, stride=2, device=device)
        layers = [
            first_conv,
            nn.ReLU(),
        ]
        k = 1
        for _ in range(1, num_layers):
            layers += [
                nn.Conv2d(channels * k, channels * (k * 2), 4, stride=2, device=device),
                nn.ReLU(),
            ]
            k = k * 2
        self.encoder = nn.Sequential(*layers)

    def forward(self, observation):
        *batch_sizes, C, H, W = observation.shape
        if len(batch_sizes) == 0:
            end_dim = 0
        else:
            end_dim = len(batch_sizes) - 1
        observation = torch.flatten(observation, start_dim=0, end_dim=end_dim)
        obs_encoded = self.encoder(observation)
        latent = obs_encoded.reshape(*batch_sizes, -1)
        return latent


class ObsDecoder(nn.Module):
    """Observation decoder network.

    Takes the deterministic state and the stochastic belief and decodes it into a pixel observation.

    Reference: https://arxiv.org/abs/1803.10122

    Args:
        channels (int, optional): Number of hidden units in the last layer.
            Defaults to 32.
        num_layers (int, optional): Depth of the network. Defaults to 4.
        kernel_sizes (int or list of int, optional): the kernel_size of each layer.
            Defaults to ``[5, 5, 6, 6]`` if num_layers if 4, else ``[5] * num_layers``.
        latent_dim (int, optional): Input dimension (state_dim + rnn_hidden_dim).
            If None, uses LazyLinear. Defaults to None for backward compatibility.
        out_channels (int, optional): Number of output channels in the final
            ConvTranspose2d layer.  Defaults to 3 (RGB).  Set to 1 for
            grayscale.
        device (torch.device, optional): Device to create the module on.
            Defaults to None (uses default device).
    """

    def __init__(
        self,
        channels=32,
        num_layers=4,
        kernel_sizes=None,
        latent_dim=None,
        out_channels=3,
        depth=None,
        device=None,
    ):
        if depth is not None:
            warnings.warn(
                f"The depth argument in {type(self)} will soon be deprecated and "
                f"used for the depth of the network instead. Please use channels "
                f"for the layer size and num_layers for the depth until depth "
                f"replaces num_layers."
            )
            channels = depth
        if num_layers < 1:
            raise RuntimeError("num_layers cannot be smaller than 1.")

        super().__init__()
        # Use explicit Linear if latent_dim provided, else LazyLinear for backward compat
        linear_out = channels * 8 * 2 * 2
        if latent_dim is not None:
            first_linear = nn.Linear(latent_dim, linear_out, device=device)
        else:
            first_linear = nn.LazyLinear(linear_out, device=device)
        self.state_to_latent = nn.Sequential(
            first_linear,
            nn.ReLU(),
        )
        if kernel_sizes is None and num_layers == 4:
            kernel_sizes = [5, 5, 6, 6]
        elif kernel_sizes is None:
            kernel_sizes = 5
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * num_layers
        layers = [
            nn.ReLU(),
            nn.ConvTranspose2d(
                channels, out_channels, kernel_sizes[-1], stride=2, device=device
            ),
        ]
        kernel_sizes = kernel_sizes[:-1]
        k = 1
        for j in range(1, num_layers):
            if j != num_layers - 1:
                layers = [
                    nn.ConvTranspose2d(
                        channels * k * 2,
                        channels * k,
                        kernel_sizes[-1],
                        stride=2,
                        device=device,
                    ),
                ] + layers
                kernel_sizes = kernel_sizes[:-1]
                k = k * 2
                layers = [nn.ReLU()] + layers
            else:
                # Use explicit ConvTranspose2d - input is always channels * 8 from state_to_latent
                layers = [
                    nn.ConvTranspose2d(
                        linear_out,
                        channels * k,
                        kernel_sizes[-1],
                        stride=2,
                        device=device,
                    )
                ] + layers

        self.decoder = nn.Sequential(*layers)
        self._depth = channels

    def forward(self, state, rnn_hidden):
        latent = self.state_to_latent(torch.cat([state, rnn_hidden], dim=-1))
        *batch_sizes, D = latent.shape
        latent = latent.view(-1, D, 1, 1)
        obs_decoded = self.decoder(latent)
        _, C, H, W = obs_decoded.shape
        obs_decoded = obs_decoded.view(*batch_sizes, C, H, W)
        return obs_decoded


class RSSMRollout(TensorDictModuleBase):
    """Rollout the RSSM network.

    Given a set of encoded observations and actions, this module will rollout the RSSM network to compute all the intermediate
    states and beliefs.
    The previous posterior is used as the prior for the next time step.
    The forward method returns a stack of all intermediate states and beliefs.

    Reference: https://arxiv.org/abs/1811.04551

    Args:
        rssm_prior (TensorDictModule): Prior network.
        rssm_posterior (TensorDictModule): Posterior network.
        use_scan (bool, optional): If True, uses torch._higher_order_ops.scan for
            the rollout loop. This is more torch.compile friendly but may have
            different performance characteristics. Defaults to False.
        compile_step (bool, optional): If True, compiles the individual step function.
            Only used when use_scan=False. Defaults to False.
        compile_backend (str, optional): Backend to use for compilation.
            Defaults to "inductor".
        compile_mode (str, optional): Mode to use for compilation.
            Defaults to None (uses PyTorch default).


    """

    def __init__(
        self,
        rssm_prior: TensorDictModule,
        rssm_posterior: TensorDictModule,
        use_scan: bool = False,
        compile_step: bool = False,
        compile_backend: str = "inductor",
        compile_mode: str | None = None,
    ):
        super().__init__()
        _module = TensorDictSequential(rssm_prior, rssm_posterior)
        self.in_keys = _module.in_keys
        self.out_keys = _module.out_keys
        self.rssm_prior = rssm_prior
        self.rssm_posterior = rssm_posterior
        self.use_scan = use_scan
        self.compile_step = compile_step
        self.compile_backend = compile_backend
        self.compile_mode = compile_mode
        self._compiled_step = None

    def _get_step_fn(self):
        """Get the step function, optionally compiled."""
        if self.compile_step and self._compiled_step is None:
            self._compiled_step = torch.compile(
                self._step,
                backend=self.compile_backend,
                mode=self.compile_mode,
            )
        return self._compiled_step if self.compile_step else self._step

    def _step(self, _tensordict):
        """Single RSSM step: prior + posterior."""
        self.rssm_prior(_tensordict)
        self.rssm_posterior(_tensordict)
        return _tensordict

    def forward(self, tensordict):
        """Runs a rollout of simulated transitions in the latent space given a sequence of actions and environment observations.

        The rollout requires a belief and posterior state primer.

        At each step, two probability distributions are built and sampled from:

        - A prior distribution p(s_{t+1} | s_t, a_t, b_t) where b_t is a
            deterministic transform of the form b_t(s_{t-1}, a_{t-1}). The
            previous state s_t is sampled according to the posterior
            distribution (see below), creating a chain of posterior-to-priors
            that accumulates evidence to compute a prior distribution over
            the current event distribution:
            p(s_{t+1} s_t | o_t, a_t, s_{t-1}, a_{t-1}) = p(s_{t+1} | s_t, a_t, b_t) q(s_t | b_t, o_t)

        - A posterior distribution of the form q(s_{t+1} | b_{t+1}, o_{t+1})
            which amends to q(s_{t+1} | s_t, a_t, o_{t+1})

        """
        if self.use_scan:
            return self._forward_scan(tensordict)
        return self._forward_loop(tensordict)

    def _forward_loop(self, tensordict):
        """Traditional loop-based forward."""
        tensordict_out = []
        *batch, time_steps = tensordict.shape

        update_values = tensordict.exclude(*self.out_keys).unbind(-1)
        _tensordict = update_values[0]
        step_fn = self._get_step_fn()

        # Determine output keys from first timestep to ensure consistent stacking.
        # Root state/belief may be added by carry_forward for t>0 but won't exist
        # for t=0, so we use the original input structure as reference.
        output_keys = list(
            update_values[0].keys(include_nested=True, leaves_only=True)
        ) + list(self.out_keys)

        for t in range(time_steps):
            _tensordict = step_fn(_tensordict)

            # Select consistent keys for stacking (excludes root state/belief
            # that may have been added by carry_forward for t>0)
            tensordict_out.append(_tensordict.select(*output_keys, strict=False))
            if t < time_steps - 1:
                # Propagate state/belief from ("next", ...) to root level for next iteration
                # The posterior outputs ("next", "state") which should become "state" for t+1
                # The prior outputs ("next", "belief") which should become "belief" for t+1
                next_state = _tensordict.get(("next", "state"))
                next_belief = _tensordict.get(("next", "belief"))

                # Get next timestep's input data (action, encoded_latents, etc.)
                _tensordict = update_values[t + 1]

                # Set the propagated state/belief (overwriting original data's initial values)
                _tensordict.set("state", next_state)
                _tensordict.set("belief", next_belief)

        out = torch.stack(tensordict_out, tensordict.ndim - 1)
        return out

    def _forward_scan(self, tensordict):
        """Scan-based forward using torch._higher_order_ops.scan.

        This is more torch.compile friendly as it avoids Python control flow.
        """
        *batch, time_steps = tensordict.shape

        update_values = tensordict.exclude(*self.out_keys).unbind(-1)
        init_td = update_values[0]

        # Determine output keys from first timestep to ensure consistent stacking.
        output_keys = list(
            update_values[0].keys(include_nested=True, leaves_only=True)
        ) + list(self.out_keys)

        # Stack the update values for scan input
        stacked_updates = torch.stack(list(update_values), dim=0)

        def scan_fn(carry, x):
            # carry is the current tensordict with propagated state/belief
            # x is the next timestep's input data (action, encoded_latents, etc.)

            # Get propagated state/belief from previous step's output
            next_state = carry.get(("next", "state"), None)
            next_belief = carry.get(("next", "belief"), None)

            # Start with next timestep's data
            _td = x

            # Propagate state/belief if available (not first step)
            if next_state is not None:
                _td.set("state", next_state)
            if next_belief is not None:
                _td.set("belief", next_belief)

            # Run prior and posterior
            self.rssm_prior(_td)
            self.rssm_posterior(_td)

            # Select consistent keys for stacking
            output_td = _td.select(*output_keys, strict=False)

            # Return output for stacking and full _td as carry for propagation
            return _td, output_td

        # Run scan
        _, outputs = _higher_order_scan(scan_fn, [init_td], [stacked_updates])

        # outputs is stacked along dim 0, move to time dimension
        out = outputs.transpose(0, tensordict.ndim - 1)
        return out


class RSSMPrior(nn.Module):
    """The prior network of the RSSM.

    This network takes as input the previous state and belief and the current action.
    It returns the next prior state and belief, as well as the parameters of the prior state distribution.
    State is by construction stochastic and belief is deterministic. In "Dream to control", these are called "deterministic state " and "stochastic state", respectively.

    Reference: https://arxiv.org/abs/1811.04551

    Args:
        action_spec (TensorSpec): Action spec.
        hidden_dim (int, optional): Number of hidden units in the linear network. Input size of the recurrent network.
            Defaults to 200.
        rnn_hidden_dim (int, optional): Number of hidden units in the recurrent network. Also size of the belief.
            Defaults to 200.
        state_dim (int, optional): Size of the state.
            Defaults to 30.
        scale_lb (:obj:`float`, optional): Lower bound of the scale of the state distribution.
            Defaults to 0.1.
        action_dim (int, optional): Dimension of the action. If provided along with state_dim,
            uses explicit Linear instead of LazyLinear. Defaults to None for backward compatibility.
        device (torch.device, optional): Device to create the module on.
            Defaults to None (uses default device).


    """

    def __init__(
        self,
        action_spec,
        hidden_dim=200,
        rnn_hidden_dim=200,
        state_dim=30,
        scale_lb=0.1,
        action_dim=None,
        device=None,
    ):
        super().__init__()

        # Prior - use explicit Linear if action_dim provided, else LazyLinear
        self.rnn = GRUCell(hidden_dim, rnn_hidden_dim, device=device)
        if action_dim is not None:
            projector_in = state_dim + action_dim
            first_linear = nn.Linear(projector_in, hidden_dim, device=device)
        else:
            first_linear = nn.LazyLinear(hidden_dim, device=device)
        self.action_state_projector = nn.Sequential(first_linear, nn.ELU())
        self.rnn_to_prior_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, device=device),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * state_dim, device=device),
            NormalParamExtractor(
                scale_lb=scale_lb,
                scale_mapping="softplus",
            ),
        )

        self.state_dim = state_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.action_shape = action_spec.shape

    def forward(self, state, belief, action, noise=None):
        """Forward pass through the prior network.

        Args:
            state: Previous stochastic state.
            belief: Previous deterministic belief.
            action: Action to condition on.
            noise: Optional pre-sampled noise for the prior state.
                If None, samples from standard normal. Used for deterministic testing.

        Returns:
            Tuple of (prior_mean, prior_std, state, belief).
        """
        projector_input = torch.cat([state, action], dim=-1)
        action_state = self.action_state_projector(projector_input)
        unsqueeze = False
        if UNSQUEEZE_RNN_INPUT and action_state.ndimension() == 1:
            if belief is not None:
                belief = belief.unsqueeze(0)
            action_state = action_state.unsqueeze(0)
            unsqueeze = True

        # GRUCell can have issues with bfloat16 autocast on some GPU/cuBLAS combinations.
        # Run the RNN in full precision to avoid CUBLAS_STATUS_INVALID_VALUE errors.
        dtype = action_state.dtype
        device_type = action_state.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            belief = self.rnn(
                action_state.float(), belief.float() if belief is not None else None
            )
        belief = belief.to(dtype)
        if unsqueeze:
            belief = belief.squeeze(0)

        prior_mean, prior_std = self.rnn_to_prior_projector(belief)
        if noise is None:
            noise = torch.randn_like(prior_std)
        state = prior_mean + noise * prior_std
        return prior_mean, prior_std, state, belief


class RSSMPosterior(nn.Module):
    """The posterior network of the RSSM.

    This network takes as input the belief and the associated encoded observation.
    It returns the parameters of the posterior as well as a state sampled according to this distribution.

    Reference: https://arxiv.org/abs/1811.04551

    Args:
        hidden_dim (int, optional): Number of hidden units in the linear network.
            Defaults to 200.
        state_dim (int, optional): Size of the state.
            Defaults to 30.
        scale_lb (:obj:`float`, optional): Lower bound of the scale of the state distribution.
            Defaults to 0.1.
        rnn_hidden_dim (int, optional): Dimension of the belief/rnn hidden state.
            If provided along with obs_embed_dim, uses explicit Linear. Defaults to None.
        obs_embed_dim (int, optional): Dimension of the observation embedding.
            If provided along with rnn_hidden_dim, uses explicit Linear. Defaults to None.
        device (torch.device, optional): Device to create the module on.
            Defaults to None (uses default device).

    """

    def __init__(
        self,
        hidden_dim=200,
        state_dim=30,
        scale_lb=0.1,
        rnn_hidden_dim=None,
        obs_embed_dim=None,
        device=None,
    ):
        super().__init__()
        # Use explicit Linear if both dims provided, else LazyLinear for backward compat
        if rnn_hidden_dim is not None and obs_embed_dim is not None:
            projector_in = rnn_hidden_dim + obs_embed_dim
            first_linear = nn.Linear(projector_in, hidden_dim, device=device)
        else:
            first_linear = nn.LazyLinear(hidden_dim, device=device)
        self.obs_rnn_to_post_projector = nn.Sequential(
            first_linear,
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * state_dim, device=device),
            NormalParamExtractor(
                scale_lb=scale_lb,
                scale_mapping="softplus",
            ),
        )
        self.hidden_dim = hidden_dim

    def forward(self, belief, obs_embedding, noise=None):
        """Forward pass through the posterior network.

        Args:
            belief: Deterministic belief from the prior.
            obs_embedding: Encoded observation.
            noise: Optional pre-sampled noise for the posterior state.
                If None, samples from standard normal. Used for deterministic testing.

        Returns:
            Tuple of (posterior_mean, posterior_std, state).
        """
        posterior_mean, posterior_std = self.obs_rnn_to_post_projector(
            torch.cat([belief, obs_embedding], dim=-1)
        )
        if noise is None:
            noise = torch.randn_like(posterior_std)
        state = posterior_mean + noise * posterior_std
        return posterior_mean, posterior_std, state
