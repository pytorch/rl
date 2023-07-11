# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Sequence, Tuple, Type, Union

import numpy as np

import torch
from torch import nn

from .models import MLP


class MultiAgentMLP(nn.Module):
    """Mult-agent MLP.

    This is an MLP that can be used in multi-agent contexts.
    For example as a policy or as a value function.
    See `examples/multiagent` for examples.

    It expects inputs with shape (*B, n_agents, n_agent_inputs)
    It returns outputs with shape (*B, n_agents, n_agent_outputs)

    If `share_params` is True, the same MLP will be used to make the forward pass for all agents (homogeneous policies).
    Otherwise, each agent will use a different MLP to process its input (heterogeneous policies).

    If `centralised` is True, each agent will use the inputs of all agents to compute its output
    (n_agent_inputs * n_agents will be the nu,ber of inputs for one agent).
    Otherwise, each agent will only use its data as input.

    **kwargs for :class:`~torchrl.modules.models.MLP` can be passed to customize the MLPs.

    """

    def __init__(
        self,
        n_agent_inputs: int,
        n_agent_outputs: int,
        n_agents: int,
        centralised: bool,
        share_params: bool,
        device: Union[torch.device, str],
        depth: Optional[int] = None,
        num_cells: Optional[Union[Sequence, int]] = None,
        activation_class: Type[nn.Module] = nn.Tanh,
        **kwargs,
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
                    **kwargs,
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
                # If the parameters are shared, and it is centralised, all agents will have the same output
                # We expand it to maintain the agent dimension, but values will be the same for all agents
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


class Mixer(nn.Module):
    """A multi-agent value mixer.

    It transforms the local value of each agent's chosen action of shape (*B, self.n_agents, 1),
    into a global value with shape (*B, 1).
    Used with the :class:`~torchrl.objectives.QMixerLoss`.
    See `examples/multiagent/qmix_vdn.py` for examples.

    Args:
        n_agents (int): number of agents,
        device (str or torch.Device): torch device for the network
        needs_state (bool): whether the mixer takes a global state as input
        state_shape (tuple or torch.Size): the shape of the state (excluding eventual leading batch dimensions)

    Examples:
        Creating a VDN mixer
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.modules.models.multiagent import VDNMixer
        >>> n_agents = 4
        >>> vdn = TensorDictModule(
        ...     module=VDNMixer(
        ...         n_agents=n_agents,
        ...         device="cpu",
        ...     ),
        ...     in_keys=[("agents","chosen_action_value")],
        ...     out_keys=["chosen_action_value"],
        ... )
        >>> td = TensorDict({"agents": TensorDict({"chosen_action_value": torch.zeros(32, n_agents, 1)}, [32, n_agents])}, [32])
        >>> td
        TensorDict(
            fields={
                agents: TensorDict(
                    fields={
                        chosen_action_value: Tensor(shape=torch.Size([32, 4, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([32, 4]),
                    device=None,
                    is_shared=False)},
            batch_size=torch.Size([32]),
            device=None,
            is_shared=False)
        >>> vdn(td)
        TensorDict(
            fields={
                agents: TensorDict(
                    fields={
                        chosen_action_value: Tensor(shape=torch.Size([32, 4, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([32, 4]),
                    device=None,
                    is_shared=False),
                chosen_action_value: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([32]),
            device=None,
            is_shared=False)


        Creating a QMix mixer
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.modules.models.multiagent import QMixer
        >>> n_agents = 4
        >>> qmix = TensorDictModule(
        ...     module=QMixer(
        ...         state_shape=(64, 64, 3),
        ...         mixing_embed_dim=32,
        ...         n_agents=n_agents,
        ...         device="cpu",
        ...     ),
        ...     in_keys=[("agents", "chosen_action_value"), "state"],
        ...     out_keys=["chosen_action_value"],
        ... )
        >>> td = TensorDict({"agents": TensorDict({"chosen_action_value": torch.zeros(32, n_agents, 1)}, [32, n_agents]), "state": torch.zeros(32, 64, 64, 3)}, [32])
        >>> td
        TensorDict(
            fields={
                agents: TensorDict(
                    fields={
                        chosen_action_value: Tensor(shape=torch.Size([32, 4, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([32, 4]),
                    device=None,
                    is_shared=False),
                state: Tensor(shape=torch.Size([32, 64, 64, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([32]),
            device=None,
            is_shared=False)
        >>> vdn(td)
        TensorDict(
            fields={
                agents: TensorDict(
                    fields={
                        chosen_action_value: Tensor(shape=torch.Size([32, 4, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([32, 4]),
                    device=None,
                    is_shared=False),
                chosen_action_value: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                state: Tensor(shape=torch.Size([32, 64, 64, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([32]),
            device=None,
            is_shared=False)
    """

    def __init__(
        self,
        n_agents: int,
        device,
        needs_state: bool,
        state_shape: Union[Tuple[int, ...], torch.Size],
    ):
        super().__init__()

        self.n_agents = n_agents
        self.device = device
        self.needs_state = needs_state
        self.state_shape = state_shape

    def forward(self, *inputs: Tuple[torch.Tensor]) -> torch.Tensor:
        """Forward pass of the mixer.

        Args:
            *inputs: The first input should be the value of the chosen action of shape (*B, self.n_agents, 1),
            representing the local q value of each agent.
            The second input (optional, used only in some mixers)
            is the shared state of all agents of shape (*B, *self.state_shape).

        Returns:
            The global value of the chosen actions obtained after mixing, with shape (*B, 1)

        """
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
                    f"Mixer network expected state with ending shape {self.state_shape},"
                    f" but got state shape {state.shape}"
                )

        if chosen_action_value.shape[-2:] != (self.n_agents, 1):
            raise ValueError(
                f"Mixer network expected chosen_action_value with last 2 dimensions {(self.n_agents,1)},"
                f" but got {chosen_action_value.shape}"
            )
        batch_dims = chosen_action_value.shape[:-2]

        if not self.needs_state:
            output = self.mix(chosen_action_value, None)
        else:
            output = self.mix(chosen_action_value, state)

        if output.shape != (*batch_dims, 1):
            raise ValueError(
                f"Mixer network expected output with same shape as input minus the multi-agent dimension,"
                f" but got {output.shape}"
            )

        return output

    def mix(self, chosen_action_value: torch.Tensor, state: torch.Tensor):
        """Forward pass for the mixer.

        Args:
            chosen_action_value: Tensor of shape [*B, n_agents]

        Returns:
            chosen_action_value: Tensor of shape [*B]
        """
        raise NotImplementedError


class VDNMixer(Mixer):
    """Mixer from https://arxiv.org/abs/1706.05296 .

    Examples:
        Creating a VDN mixer
         >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.modules.models.multiagent import VDNMixer
        >>> n_agents = 4
        >>> vdn = TensorDictModule(
        ...     module=VDNMixer(
        ...         n_agents=n_agents,
        ...         device="cpu",
        ...     ),
        ...     in_keys=[("agents","chosen_action_value")],
        ...     out_keys=["chosen_action_value"],
        ... )
        >>> td = TensorDict({"agents": TensorDict({"chosen_action_value": torch.zeros(32, n_agents, 1)}, [32, n_agents])}, [32])
        >>> td
        TensorDict(
            fields={
                agents: TensorDict(
                    fields={
                        chosen_action_value: Tensor(shape=torch.Size([32, 4, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([32, 4]),
                    device=None,
                    is_shared=False)},
            batch_size=torch.Size([32]),
            device=None,
            is_shared=False)
        >>> vdn(td)
        TensorDict(
            fields={
                agents: TensorDict(
                    fields={
                        chosen_action_value: Tensor(shape=torch.Size([32, 4, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([32, 4]),
                    device=None,
                    is_shared=False),
                chosen_action_value: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([32]),
            device=None,
            is_shared=False)
    """

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
        return chosen_action_value.sum(dim=-2)


class QMixer(Mixer):
    """Mixer from https://arxiv.org/abs/1803.11485 .

    Examples:
        Creating a QMix mixer
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.modules.models.multiagent import QMixer
        >>> n_agents = 4
        >>> qmix = TensorDictModule(
        ...     module=QMixer(
        ...         state_shape=(64, 64, 3),
        ...         mixing_embed_dim=32,
        ...         n_agents=n_agents,
        ...         device="cpu",
        ...     ),
        ...     in_keys=[("agents", "chosen_action_value"), "state"],
        ...     out_keys=["chosen_action_value"],
        ... )
        >>> td = TensorDict({"agents": TensorDict({"chosen_action_value": torch.zeros(32, n_agents, 1)}, [32, n_agents]), "state": torch.zeros(32, 64, 64, 3)}, [32])
        >>> td
        TensorDict(
            fields={
                agents: TensorDict(
                    fields={
                        chosen_action_value: Tensor(shape=torch.Size([32, 4, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([32, 4]),
                    device=None,
                    is_shared=False),
                state: Tensor(shape=torch.Size([32, 64, 64, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([32]),
            device=None,
            is_shared=False)
        >>> vdn(td)
        TensorDict(
            fields={
                agents: TensorDict(
                    fields={
                        chosen_action_value: Tensor(shape=torch.Size([32, 4, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([32, 4]),
                    device=None,
                    is_shared=False),
                chosen_action_value: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                state: Tensor(shape=torch.Size([32, 64, 64, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([32]),
            device=None,
            is_shared=False)
    """

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
        bs = chosen_action_value.shape[:-2]
        state = state.view(-1, self.state_dim)
        chosen_action_value = chosen_action_value.view(-1, 1, self.n_agents)
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
