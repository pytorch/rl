from typing import Tuple

import torch
from torch import nn

from torchrl.modules import MLP


class MultiAgentMLP(nn.Module):
    def __init__(
        self,
        n_agent_inputs,
        n_agent_outputs,
        n_agents,
        centralised,
        share_params,
        device,
        depth,
        num_cells,
        activation_class,
    ):
        super().__init__()

        self.n_agents = n_agents
        self.n_agent_inputs = n_agent_inputs
        self.n_agent_outputs = n_agent_outputs
        self.share_params = share_params
        self.centralised = centralised

        self.agent_networks = nn.ModuleList(
            [
                MLP(
                    in_features=n_agent_inputs
                    if not centralised
                    else n_agent_inputs * n_agents,
                    out_features=n_agent_outputs,
                    depth=depth,
                    num_cells=num_cells,
                    activation_class=activation_class,
                    device=device,
                )
                for _ in range(self.n_agents if not self.share_params else 1)
            ]
        )

    def forward(self, *inputs: Tuple[torch.Tensor]) -> torch.Tensor:
        if len(inputs) > 1:
            inputs = (torch.cat([*inputs], -1),)
        inputs = inputs[0]

        if inputs.shape[-2:] != (self.n_agents, self.n_agent_inputs):
            raise ValueError(
                f"Multi-agent network expected input with last 2 dimensions {[self.n_agents, self.n_agent_inputs]},"
                f" but got {inputs.shape}"
            )

        # If the model is centralized, agents have full observability
        if self.centralised:
            inputs = inputs.view(
                *inputs.shape[:-2], self.n_agents * self.n_agent_inputs
            )

        # If parameters are not shared, each agent has its own network
        if not self.share_params:
            if self.centralised:
                output = torch.stack(
                    [net(inputs) for i, net in enumerate(self.agent_networks)],
                    dim=-2,
                )
            else:
                output = torch.stack(
                    [
                        net(inputs[..., i, :])
                        for i, net in enumerate(self.agent_networks)
                    ],
                    dim=-2,
                )
        # If parameters are shared, agents use the same network
        else:
            output = self.agent_networks[0](inputs)

            if self.centralised:
                output = (
                    output.view(*output.shape[:-1], self.n_agent_outputs)
                    .unsqueeze(-2)
                    .expand(*output.shape[:-1], self.n_agents, self.n_agent_outputs)
                )

        if output.shape[-2:] != (self.n_agents, self.n_agent_outputs):
            raise ValueError(
                f"Multi-agent network expected output with last 2 dimensions {[self.n_agents, self.n_agent_outputs]},"
                f" but got {output.shape}"
            )

        return output
