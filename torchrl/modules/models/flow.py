# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import torch
from torch import nn, Tensor

from torchrl._utils import implement_for, is_compiling

__all__ = ["FlowMatchingModel", "OneStepModel"]


class FlowMatchingModel(nn.Module):
    """Tensor-only Euler sampler for bounded continuous actions.

    Args:
        velocity_network (nn.Module): maps concatenated observation, action and
            scalar time to an action-sized velocity.
        action_dim (int): number of action coordinates.
        num_steps (int, optional): Euler integration steps. Defaults to 10.

    Keyword Args:
        unroll (int, optional): Euler steps per scan body on PyTorch 2.14+.
            Defaults to 1. Larger values trade graph size for fewer iterations.
            Ignored by the explicit-loop compatibility paths.
        low (float or Tensor, optional): lower action bound, broadcast over
            actions. Must be strictly less than ``high``. Defaults to -1.
        high (float or Tensor, optional): upper action bound, broadcast over
            actions. Defaults to 1.

    Outputs are clipped to ``[low, high]`` after integration.
    :class:`~torchrl.modules.FlowMatchingPolicy` provides the TensorDict interface.
    On PyTorch 2.14+, eager execution (including autograd) and compiled
    inference use scan. Older versions and compiled calls with gradients enabled
    use an explicit Euler loop: PyTorch 2.14 Inductor can produce incorrect
    gradients through a peeled scan followed by clipping.
    """

    def __init__(
        self,
        velocity_network: nn.Module,
        action_dim: int,
        num_steps: int = 10,
        *,
        unroll: int = 1,
        low: float | Tensor = -1.0,
        high: float | Tensor = 1.0,
    ) -> None:
        super().__init__()
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        if unroll < 1:
            raise ValueError("unroll must be positive")
        self.unroll = unroll
        self.velocity_network = velocity_network
        self.action_dim = action_dim
        self.num_steps = num_steps
        self.register_buffer("low", torch.as_tensor(low))
        self.register_buffer("high", torch.as_tensor(high))
        if not (self.low < self.high).all():
            raise ValueError("low must be strictly less than high")

    def velocity(self, observation: Tensor, action: Tensor, time: Tensor) -> Tensor:
        return self.velocity_network(torch.cat((observation, action, time), -1))

    def forward(self, observation: Tensor, noise: Tensor | None = None) -> Tensor:
        if noise is None:
            noise = observation.new_empty(
                (*observation.shape[:-1], self.action_dim)
            ).normal_()
        action = self.integrate(observation, noise)
        return action.clamp(self.low.to(action), self.high.to(action))

    def euler(self, observation: Tensor, action: Tensor) -> Tensor:
        for step in range(self.num_steps):
            time = action.new_full((*action.shape[:-1], 1), step / self.num_steps)
            action = action + self.velocity(observation, action, time) / self.num_steps
        return action

    @implement_for("torch", None, "2.14", compilable=True)
    def integrate(self, observation: Tensor, action: Tensor) -> Tensor:
        return self.euler(observation, action)

    @implement_for("torch", "2.14", compilable=True)
    def integrate(self, observation: Tensor, action: Tensor) -> Tensor:  # noqa: F811
        # Inductor can corrupt gradients through the peeled scan and final clamp.
        if is_compiling() and torch.is_grad_enabled():
            return self.euler(observation, action)
        num_steps = self.num_steps
        unroll = min(self.unroll, num_steps)

        def euler_step(action: Tensor, time: Tensor) -> Tensor:
            time = time.to(action).expand(*action.shape[:-1], 1)
            return action + self.velocity(observation, action, time) / num_steps

        def step_block(action: Tensor, times: Tensor) -> tuple[Tensor, Tensor]:
            for time in times.unbind(0):
                action = euler_step(action, time)
            # Scan forbids aliasing between its carry and stacked outputs.
            return action, action.clone()

        # The first update establishes the scan carry dtype and layout.
        # Peeling the remainder also keeps every scan block the same length.
        start = 1 + (num_steps - 1) % unroll
        for step in range(start):
            action = euler_step(action, action.new_tensor(step / num_steps))
        time_dtype = torch.promote_types(action.dtype, torch.float32)
        times = torch.arange(start, num_steps, device=action.device, dtype=time_dtype)
        action, _ = torch._higher_order_ops.scan(
            step_block, action, (times / num_steps).reshape(-1, unroll)
        )
        return action


class OneStepModel(nn.Module):
    """Tensor-only network for one-step flow distillation.

    Args:
        network (nn.Module): maps concatenated observation and Gaussian noise
            directly to an action, without an output activation.
        action_dim (int): number of action coordinates.

    Keyword Args:
        low (float or Tensor, optional): lower action bound, broadcast over
            actions. Must be strictly less than ``high``. Defaults to -1.
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
        if not (self.low < self.high).all():
            raise ValueError("low must be strictly less than high")

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
