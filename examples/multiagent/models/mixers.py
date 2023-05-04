from typing import Tuple

import numpy as np
import torch
from torch import nn


class Mixer(nn.Module):
    def __init__(
        self,
        n_agents: int,
        device,
        needs_state: bool,
        state_shape: torch.Size,
    ):
        super().__init__()

        self.n_agents = n_agents
        self.device = device
        self.needs_state = needs_state
        self.state_shape = state_shape

    def forward(self, *inputs: Tuple[torch.Tensor]) -> torch.Tensor:
        if not self.needs_state:
            if len(inputs) > 1:
                raise ValueError(
                    "Mixer that doesn't need state was passed more than 1 input"
                )
            chosen_action_value = inputs[0]
        else:
            if len(inputs) > 2:
                raise ValueError("Mixer that needs state was passed more than 2 inputs")

            chosen_action_value, state = inputs

            if state.shape[-len(self.state_shape) :] != self.state_shape:
                raise ValueError(
                    f"Mixer network expected state with ending shape {self.state_shape} in penultimate dimension,"
                    f" but got state shape {state.shape}"
                )

        if chosen_action_value.shape[-2:] != (self.n_agents, 1):
            raise ValueError(
                f"Mixer network expected chosen_action_value with last 2 dimensions {[self.n_agents, 1]},"
                f" but got {chosen_action_value.shape}"
            )

        if not self.needs_state:
            output = self.mix(chosen_action_value, None)
        else:
            output = self.mix(chosen_action_value, state)

        output = (
            output.view(*output.shape[:-1], 1)
            .unsqueeze(-2)
            .expand(*output.shape[:-1], self.n_agents, 1)
        )

        if output.shape[-2:] != (self.n_agents, 1):
            raise ValueError(
                f"Mixer network expected output with last 2 dimensions {[self.n_agents, 1]},"
                f" but got {output.shape}"
            )

        return output

    def mix(self, chosen_action_value: torch.Tensor, state: torch.Tensor):
        raise NotImplementedError


class VDNMixer(Mixer):
    def __init__(
        self,
        n_agents: int,
        device,
    ):
        super().__init__(
            needs_state=False,
            state_shape=torch.Size([]),
            n_agents=n_agents,
            device=device,
        )

    def mix(self, chosen_action_value: torch.Tensor, state: torch.Tensor):
        """Forward pass for the mixer.

        Args:
            chosen_action_value: Tensor of shape [*B, n_agents, 1]
        """
        return chosen_action_value.sum(dim=-2)


class QMixer(Mixer):
    def __init__(
        self,
        state_shape,
        mixing_embed_dim,
        n_agents: int,
        device,
    ):
        super().__init__(
            needs_state=True, state_shape=state_shape, n_agents=n_agents, device=device
        )

        self.embed_dim = mixing_embed_dim
        self.state_dim = int(np.prod(state_shape))

        self.hyper_w_1 = nn.Linear(
            self.state_dim, self.embed_dim * self.n_agents, device=self.device
        )
        self.hyper_w_final = nn.Linear(
            self.state_dim, self.embed_dim, device=self.device
        )

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim, device=self.device)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim, device=self.device),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1, device=self.device),
        )

    def mix(self, chosen_action_value: torch.Tensor, state: torch.Tensor):
        """Forward pass for the mixer.

        Args:
            chosen_action_value: Tensor of shape [*B, n_agents, 1]
            state: Tensor of shape [*B, *state_shape]
        """
        bs = chosen_action_value.shape[:-2]
        state = state.view(-1, self.state_dim)
        chosen_action_value = chosen_action_value.transpose(-2, -1).view(
            -1, 1, self.n_agents
        )
        # First layer
        w1 = torch.abs(self.hyper_w_1(state))
        b1 = self.hyper_b_1(state)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = nn.functional.elu(
            torch.bmm(chosen_action_value, w1) + b1
        )  # [-1, 1, self.embed_dim]
        # Second layer
        w_final = torch.abs(self.hyper_w_final(state))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(state).view(-1, 1, 1)
        # Compute final output
        y = torch.bmm(hidden, w_final) + v  # [-1, 1, 1]
        # Reshape and return
        q_tot = y.view(*bs, 1)
        return q_tot
