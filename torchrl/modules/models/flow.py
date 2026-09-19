# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import torch
from torch import nn


class FlowMatchingPolicy(nn.Module):
    """Conditional flow policy for normalized continuous actions.

    Args:
        velocity_network (nn.Module): maps concatenated observation, action and
            scalar time to an action-sized velocity.
        action_dim (int): number of action coordinates.
        num_steps (int, optional): Euler integration steps. Defaults to 10.

    Outputs are clipped to ``[-1, 1]`` after integration. Wrap in
    :class:`~torchrl.modules.Actor` for TensorDict and collector support.
    """

    def __init__(
        self, velocity_network: nn.Module, action_dim: int, num_steps: int = 10
    ):
        super().__init__()
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        self.velocity_network = velocity_network
        self.action_dim = action_dim
        self.num_steps = num_steps

    def velocity(self, observation, action, time):
        return self.velocity_network(torch.cat((observation, action, time), -1))

    def forward(self, observation, noise=None):
        if noise is None:
            noise = observation.new_empty(
                (*observation.shape[:-1], self.action_dim)
            ).normal_()
        action = noise
        for step in range(self.num_steps):
            time = action.new_full((*action.shape[:-1], 1), step / self.num_steps)
            action = action + self.velocity(observation, action, time) / self.num_steps
        return action.clamp(-1, 1)


class OneStepPolicy(nn.Module):
    """Noise-conditioned policy distilled from a flow policy.

    Args:
        network (nn.Module): maps concatenated observation and Gaussian noise
            directly to an action, without an output activation.
        action_dim (int): number of action coordinates.

    Outputs are clipped to ``[-1, 1]``. Wrap in :class:`~torchrl.modules.Actor`
    for TensorDict and collector support. Both policies sample Gaussian noise
    even under deterministic exploration; pass explicit noise for repeatability.
    """

    def __init__(self, network: nn.Module, action_dim: int):
        super().__init__()
        self.network = network
        self.action_dim = action_dim

    def forward(self, observation, noise=None, *, clamp=True):
        if noise is None:
            noise = observation.new_empty(
                (*observation.shape[:-1], self.action_dim)
            ).normal_()
        action = self.network(torch.cat((observation, noise), -1))
        return action.clamp(-1, 1) if clamp else action
