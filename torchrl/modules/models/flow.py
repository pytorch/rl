# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn, Tensor

from torchrl._utils import implement_for, is_compiling

__all__ = ["FlowMatchingModel", "OneStepModel"]


@implement_for("torch", None, "2.14")
def get_flow_scan() -> Callable | None:
    return None


@implement_for("torch", "2.14")
def get_flow_scan() -> Callable | None:  # noqa: F811
    return torch._higher_order_ops.scan


FLOW_SCAN = get_flow_scan()


class FlowMatchingModel(nn.Module):
    """Tensor-only Euler sampler for bounded continuous actions.

    Args:
        velocity_network (nn.Module): maps concatenated observation, action and
            scalar time to an action-sized velocity.
        action_dim (int): number of action coordinates.
        num_steps (int, optional): Euler integration steps. Defaults to 10.

    Keyword Args:
        low (float or Tensor, optional): lower action bound, broadcast over
            actions. Defaults to -1.
        high (float or Tensor, optional): upper action bound, broadcast over
            actions. Defaults to 1.

    Outputs are clipped to ``[low, high]`` after integration.
    :class:`~torchrl.modules.FlowMatchingPolicy` provides the TensorDict interface.
    On PyTorch 2.14+, compiled sampling without gradients uses a scan loop.
    """

    def __init__(
        self,
        velocity_network: nn.Module,
        action_dim: int,
        num_steps: int = 10,
        *,
        low: float | Tensor = -1.0,
        high: float | Tensor = 1.0,
    ) -> None:
        super().__init__()
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        self.velocity_network = velocity_network
        self.action_dim = action_dim
        self.num_steps = num_steps
        self.register_buffer("low", torch.as_tensor(low))
        self.register_buffer("high", torch.as_tensor(high))

    def velocity(self, observation: Tensor, action: Tensor, time: Tensor) -> Tensor:
        return self.velocity_network(torch.cat((observation, action, time), -1))

    def forward(self, observation: Tensor, noise: Tensor | None = None) -> Tensor:
        if noise is None:
            noise = observation.new_empty(
                (*observation.shape[:-1], self.action_dim)
            ).normal_()
        num_steps = self.num_steps
        if (
            FLOW_SCAN is not None
            and is_compiling()
            and not torch.is_grad_enabled()
            and num_steps > 1
        ):

            def euler_step(action: Tensor, time: Tensor) -> tuple[Tensor, Tensor]:
                time = time.to(action).expand(*action.shape[:-1], 1)
                action = action + self.velocity(observation, action, time) / num_steps
                return action, action.clone()

            # The first update establishes the scan carry dtype and layout.
            action, _ = euler_step(noise, noise.new_zeros(()))
            time_dtype = torch.promote_types(action.dtype, torch.float32)
            times = torch.arange(1, num_steps, device=action.device, dtype=time_dtype)
            action, _ = FLOW_SCAN(euler_step, action, times / num_steps)
        else:
            action = noise
            for step in range(num_steps):
                time = action.new_full((*action.shape[:-1], 1), step / num_steps)
                action = action + self.velocity(observation, action, time) / num_steps
        return action.clamp(self.low.to(action), self.high.to(action))


class OneStepModel(nn.Module):
    """Tensor-only network for one-step flow distillation.

    Args:
        network (nn.Module): maps concatenated observation and Gaussian noise
            directly to an action, without an output activation.
        action_dim (int): number of action coordinates.

    Keyword Args:
        low (float or Tensor, optional): lower action bound, broadcast over
            actions. Defaults to -1.
        high (float or Tensor, optional): upper action bound, broadcast over
            actions. Defaults to 1.

    :class:`~torchrl.modules.OneStepPolicy` provides the TensorDict interface.
    Outputs are clipped to ``[low, high]``. Both models sample Gaussian noise
    even under deterministic exploration; pass explicit noise for repeatability.
    """

    def __init__(
        self,
        network: nn.Module,
        action_dim: int,
        *,
        low: float | Tensor = -1.0,
        high: float | Tensor = 1.0,
    ) -> None:
        super().__init__()
        self.network = network
        self.action_dim = action_dim
        self.register_buffer("low", torch.as_tensor(low))
        self.register_buffer("high", torch.as_tensor(high))

    def forward(
        self, observation: Tensor, noise: Tensor | None = None, *, clamp: bool = True
    ) -> Tensor:
        if noise is None:
            noise = observation.new_empty(
                (*observation.shape[:-1], self.action_dim)
            ).normal_()
        action = self.network(torch.cat((observation, noise), -1))
        return (
            action.clamp(self.low.to(action), self.high.to(action)) if clamp else action
        )
